import torch.nn.functional as F
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger 
import os
import torch
import yaml
import argparse
from torch_geometric.data import Batch
from models.ginet_finetune import GINet
from dataset.dataset_test import MolTestDatasetWrapper
from dataset.dataset_test import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST
from torch_geometric.data import Data, Dataset, DataLoader

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(checkpoint_dir, device='cpu'):
    config_path = os.path.join(checkpoint_dir, 'config_finetune.yaml')
    checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')

    config = load_config(config_path)
    model_cfg = config['model']

    model = GINet(
        num_layer=model_cfg['num_layer'],
        emb_dim=model_cfg['emb_dim'],
        feat_dim=model_cfg['feat_dim'],
        drop_ratio=model_cfg.get('drop_ratio', 0),
        pool=model_cfg.get('pool', 'mean')
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def predict_bbbp(smiles_list, model, config, device='cpu'):

    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, smiles_list):
            self.smiles_list = smiles_list

        def __len__(self):
            return len(self.smiles_list)

        def __getitem__(self, idx):
            mol = Chem.MolFromSmiles(self.smiles_list[idx])
            if mol is None:
                raise ValueError(f"Invalid SMILES: {self.smiles_list[idx]}")

            type_idx = [ATOM_LIST.index(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
            chirality_idx = [CHIRALITY_LIST.index(atom.GetChiralTag()) for atom in mol.GetAtoms()]
            x = torch.tensor(list(zip(type_idx, chirality_idx)), dtype=torch.long)

            row, col, edge_feat = [], [], []
            for bond in mol.GetBonds():
                s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [s, e]
                col += [e, s]
                ef = [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir())
                ]
                edge_feat.append(ef)
                edge_feat.append(ef)

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = torch.tensor(edge_feat, dtype=torch.long)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    dataset = InferenceDataset(smiles_list)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, out = model(batch)
            preds.append(out.cpu())

    return torch.cat(preds, dim=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Currently, only BBBP is supported.")
    parser.add_argument("--task", required=True, type=str,
                        help="Task name, e.g. 'bbbp'")
    parser.add_argument("--smiles", required=True, nargs="+",
                        help="One or more SMILES strings (space-separated).")


    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    checkpoint_dir = './finetune/gin/BBBP/checkpoints/'

    model, config = load_model(checkpoint_dir, device=device)

    # TASK ROUTING
    if args.task.lower() == "bbbp":
        probs = predict_bbbp(args.smiles, model, config, device=device)
        probs_softmax = F.softmax(probs, dim=1)
        bbbp_probs = probs_softmax[:, 1]

        for smi, prob in zip(args.smiles, bbbp_probs):
            print(f"SMILES: {smi} --> BBBP probability: {prob.item():.4f}")

    else:
        raise ValueError(f"Unknown task: {args.task}")
