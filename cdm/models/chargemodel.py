import os
import torch
from ase import Atoms
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
from tqdm import tqdm
import warnings

from torch_geometric.data import Batch
from torch_geometric.utils import remove_isolated_nodes

from ocpmodels.datasets import data_list_collater
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.common.utils import pyg2_data_transform
from ocpmodels.common.utils import load_state_dict

from cdm.utils.probe_graph import ProbeGraphAdder

class customTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        ex = torch.exp(input)
        ex_1 = torch.exp(-input)
        return ((ex - ex_1)) / ((2*ex) + ex_1)


@registry.register_model('charge_model')
class ChargeModel(torch.nn.Module):
    def __init__(
        self,
        atom_model_config,
        probe_model_config,
        otf_pga_config = {
            'implementation': 'RGPBC',
        },
        include_atomic_edges = False,
        enforce_zero_for_disconnected_probes = False,
        enforce_charge_conservation = False,
        freeze_atomic = False,
        name = 'charge_model',
    ):
        super().__init__()
        
        self.regress_forces = False
        self.enforce_zero_for_disconnected_probes = enforce_zero_for_disconnected_probes
        self.enforce_charge_conservation = enforce_charge_conservation
        self.freeze_atomic = freeze_atomic
        
        probe_final_mlp = True
            
        # Initialize atom message-passing model
        if 'checkpoint' in atom_model_config:
            cfg = torch.load(
                atom_model_config['checkpoint'],
                map_location=torch.device('cpu')
            )['config']['model_attributes']
        else:
            cfg = atom_model_config

        
        self.atom_message_model = registry.get_model_class(atom_model_config['name'])(
            **cfg,
            atomic=True, 
            probe=False,
        )
        
        if 'checkpoint' in atom_model_config:
            self.load_checkpoint(
                checkpoint_path = atom_model_config['checkpoint'],
                atomic = True,
            )
        
        
        # Initialize probe message-passing model
        if 'checkpoint' in probe_model_config:
            cfg = torch.load(
                probe_model_config['checkpoint'],
                map_location=torch.device('cpu')
            )['config']['model_attributes']
        else:
            cfg = probe_model_config

        
        self.probe_message_model = registry.get_model_class(probe_model_config['name'])(
            **cfg,
            atomic=False, 
            probe=True,
        )
        
        if 'checkpoint' in probe_model_config:
            self.load_checkpoint(
                checkpoint_path = probe_model_config['checkpoint'],
                probe = True,
            )
        
        # Ensure match between atom and probe messaging models
        if self.atom_message_model.hidden_channels != self.probe_message_model.hidden_channels:
            self.reduce_atom_representations = True
            self.atom_reduction = torch.nn.Sequential(
                torch.nn.Linear(self.atom_message_model.hidden_channels,self.atom_message_model.hidden_channels),
                torch.nn.Sigmoid(),
                torch.nn.Linear(self.atom_message_model.hidden_channels, self.probe_message_model.hidden_channels))
        else:
            self.reduce_atom_representations = False
        
        assert self.atom_message_model.num_interactions >= self.probe_message_model.num_interactions
        
        # Compatibility for specific models
        if probe_model_config['name'] == 'scn_charge':
            probe_final_mlp = False
        
        if probe_final_mlp:
           self.probe_output_function_0 = torch.nn.Sequential(
                
                torch.nn.Linear(self.probe_message_model.hidden_channels, self.probe_message_model.hidden_channels),
                #torch.nn.BatchNorm1d(self.probe_message_model.hidden_channels),
                torch.nn.ELU(),
                torch.nn.Dropout(0.2),
                
                torch.nn.Linear(self.probe_message_model.hidden_channels, self.probe_message_model.hidden_channels),
                #torch.nn.BatchNorm1d(self.probe_message_model.hidden_channels),
                torch.nn.ELU(),
                torch.nn.Dropout(0.2),
                
                torch.nn.Linear(self.probe_message_model.hidden_channels, self.probe_message_model.hidden_channels),
                #torch.nn.BatchNorm1d(self.probe_message_model.hidden_channels),
                torch.nn.ELU(),
                torch.nn.Dropout(0.2),
                
                torch.nn.Linear(self.probe_message_model.hidden_channels, 1),
                torch.nn.Softplus()
            )
 
           self.probe_output_function_1 = torch.nn.Sequential(
                
                torch.nn.Linear(self.probe_message_model.hidden_channels, self.probe_message_model.hidden_channels),
                #torch.nn.BatchNorm1d(self.probe_message_model.hidden_channels),
                torch.nn.ELU(),
                torch.nn.Dropout(0.2),
                
                torch.nn.Linear(self.probe_message_model.hidden_channels, self.probe_message_model.hidden_channels),
                #torch.nn.BatchNorm1d(self.probe_message_model.hidden_channels),
                torch.nn.ELU(),
                torch.nn.Dropout(0.2),
                
                torch.nn.Linear(self.probe_message_model.hidden_channels, self.probe_message_model.hidden_channels),
                #torch.nn.BatchNorm1d(self.probe_message_model.hidden_channels),
                torch.nn.ELU(),
                torch.nn.Dropout(0.2),
                
                torch.nn.Linear(self.probe_message_model.hidden_channels, 1),
                torch.nn.Softplus()
            )

           self.probe_output_function_2 = torch.nn.Sequential(

                torch.nn.Linear(self.probe_message_model.hidden_channels, self.probe_message_model.hidden_channels),
                #torch.nn.BatchNorm1d(self.probe_message_model.hidden_channels),
                torch.nn.ELU(),
                torch.nn.Dropout(0.2),

                torch.nn.Linear(self.probe_message_model.hidden_channels, self.probe_message_model.hidden_channels),
                #torch.nn.BatchNorm1d(self.probe_message_model.hidden_channels),
                torch.nn.ELU(),
                torch.nn.Dropout(0.2),

                torch.nn.Linear(self.probe_message_model.hidden_channels, self.probe_message_model.hidden_channels),
                #torch.nn.BatchNorm1d(self.probe_message_model.hidden_channels),
                torch.nn.ELU(),
                torch.nn.Dropout(0.2),

                torch.nn.Linear(self.probe_message_model.hidden_channels, 1),
                torch.nn.Softplus()
            )


        self.otf_pga = ProbeGraphAdder(**otf_pga_config)
        
        
    @conditional_grad(torch.enable_grad())
    def forward(self, data, task_id):
        # Ensure data has probe points
        data = self.otf_pga(data)
        data.probe_data = [pyg2_data_transform(data.probe_data)]
        data.probe_data = Batch.from_data_list(data.probe_data)
        
        atom_representations = self.forward_atomic(data)

        probes = self.forward_probe(data.probe_data, atom_representations, task_id)
        
        return probes
    
    @conditional_grad(torch.enable_grad())
    def forward_atomic(self, data):
        if self.freeze_atomic:
            with torch.no_grad():
                atom_representations = self.atom_message_model(data)
        else:
            atom_representations = self.atom_message_model(data)
        
        if self.reduce_atom_representations:
            atom_representations = [self.atom_reduction(rep).float() for rep in atom_representations]
            
        return(atom_representations)
            
    @conditional_grad(torch.enable_grad())        
    def forward_probe(self, data, atom_representations, task_id):
        data.atom_representations = atom_representations[-self.probe_message_model.num_interactions:]
        
        probe_results = self.probe_message_model(data)
        
        if hasattr(self, 'probe_output_function_0') or hasattr(self, 'probe_output_function_1') or hasattr(self, 'probe_output_function_2'):
            if task_id == 0:
                probe_results = self.probe_output_function_0(probe_results).flatten()
            elif task_id == 1:
                probe_results = self.probe_output_function_1(probe_results).flatten()
            elif task_id == 2:
                probe_results = self.probe_output_function_2(probe_results).flatten()

        probe_results = torch.nan_to_num(probe_results)
        
        if self.enforce_zero_for_disconnected_probes:
            is_probe = data.atomic_numbers == 0
            _, _, is_not_isolated = remove_isolated_nodes(data.edge_index, num_nodes = len(data.atomic_numbers))
            is_isolated = ~is_not_isolated
            
            if torch.all(is_isolated):
                warnings.warn('All probes are isolated - not enforcing zero charge constraint')
            else:
                probe_results[is_isolated[is_probe]] = torch.zeros_like(probe_results[is_isolated[is_probe]])

        if self.enforce_charge_conservation: 
            if torch.sum(probe_results) == 0:
                warnings.warn('Charge prediction is 0 - cannot enforce charge conservation!')
            else:
                data.total_target = data.total_target.to(probe_results.device)
                probe_results *= data.total_target / torch.sum(probe_results)
        
        return probe_results
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    
    def load_checkpoint(
        self,
        checkpoint_path,
        atomic=False,
        probe=False
    ):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", checkpoint_path
            )

        if atomic:
            model = self.atom_message_model
        if probe:
            model = self.probe_message_model
        
        map_location = torch.device("cpu") if self.cpu else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Match the "module." count in the keys of model and checkpoint state_dict
        # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
        # Not using either of the above two would have no "module."

        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
        mod_key_count = next(iter(model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            new_dict = checkpoint["state_dict"]
        
        # HACK TO LOAD OLD SCHNET CHECKPOINTS
        if 'atomic_mass' in new_dict:
            del new_dict['atomic_mass']
        
        load_state_dict(model, new_dict, strict=False)
