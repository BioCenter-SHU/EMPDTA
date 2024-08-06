import os
import json
import torch
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import pickle
from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


@R.register("datasets.PDBBind")
@utils.copy_args(data.ProteinLigandDataset.load_sequence)
class PDBBind(data.ProteinLigandDataset):
    """
    The PDBbind-2020 dataset with 19347 binding affinity indicating the interaction strength 
    between pairs of protein and ligand.

    Statistics:
        - #Train: 18022
        - #Valid: 962
        - #Test: 363
    Parameters:
        path (str): path to store the dataset 
        drug_method (str): Drug loading method, from 'smile','2d' or '3d'. 
        protein_method (str): Protein loading method, from 'sequence','pdb'.
        transform (Callable, optional): protein sequence transformation function
        lazy (bool, optional): if lazy mode is used, the protein-ligand pairs are processed in the dataloader.
            This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
        **kwargs
    """

    splits = ["train", "valid", "test"]
    target_fields = ["affinity"]

    def __init__(self, path='../../data/dta-datasets/PDBbind/', drug_method='3d', protein_method='sequence',
                 transform=None, lazy=False, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.transform = transform
        self.lazy = lazy
        self.protein_method = protein_method  # added for get_item method
        local_file = self.path + 'pdbbind_datasets.csv'
        # TankBind [17787, 968, 363] 19118
        self.num_samples = [18025, 962, 363]  # 19350
        dataset_df = pd.read_csv(local_file) 

        # ============================== Get the col info of the Dataframe ============================== #
        drug_list = dataset_df["Drug_Index"].tolist() # for further index the drug in 68 durg list
        protein_list = dataset_df["Drug_Index"].tolist() # dataset_df["Protein_Index"].tolist()  # for further index the protein in 442 protein list
        self.smiles = dataset_df["Drug"].tolist()
        self.targets = {}  # follow the torchdrug form create a dict to store
        self.targets["affinity"] = dataset_df["Y"].tolist()  # scaled to [0, 1]

        # ============================== Loading label file ==============================  #
        label_pkl = path + 'gearnet_labeled.pkl'
        with utils.smart_open(label_pkl, "rb") as fin:
            self.label_list = pickle.load(fin)

        # ============================== Loading ligand position file ==============================  #
        # coords_pkl = path + '3d_Molecule.pkl'
        # with utils.smart_open(coords_pkl, "rb") as fin:
        #       coords_list = pickle.load(fin)
        # ============================== Generating the self.data list [protein, mol] ============================== #
        num_sample = len(self.smiles)                
        for field, target_list in self.targets.items():
            if len(target_list) != num_sample:
                raise ValueError("Number of target `%s` doesn't match with number of molecules. "
                                 "Expect %d but found %d" % (field, num_sample, len(target_list)))
        
        self.data = []
        print(f'==================== Using {drug_method} drug method and {protein_method} protein method! ====================')
        # ============================== Loading Protein ============================== #
        # protein_pkl = path + protein_method + '_Protein.pkl'
        # # ============================== Pkl file not exist, Creating one ==============================  #
        # if not os.path.exists(protein_pkl):
        #     protein_file = self.path + 'pdbbind_proteins.csv'
        #     # 'gearnet' or 'comenet' are pre-defined through the graph construction
        #     if protein_method == 'pdb' or 'sequence' or 'gearnet' or 'comenet':
        #         self.load_protein(protein_file, protein_method, **kwargs)
        #     else:
        #         raise ValueError("Protein method should be 'pdb', 'sequence' 'gearnet' or 'comenet'!")
        #
        # # ============================== Loading Pkl file ==============================  #
        # with utils.smart_open(protein_pkl, "rb") as fin:
        #     target_protein  = pickle.load(fin)
        # ============================== Loading Drug ============================== #
        drug_pkl = path + drug_method + '_Molecule.pkl'
        # ============================== Pkl file not exist, Creating one ==============================  #
        if not os.path.exists(drug_pkl):
            drug_file = self.path + 'pdbbind_ligands.csv'
            if drug_method == 'smile' or '2d' or '3d' or 'comenet':
                self.load_drug(drug_file, drug_method, **kwargs)
            else:
                raise ValueError("Drug method should be 'smile' , '2d' '3d' or 'comenet' !")
        with utils.smart_open(drug_pkl, "rb") as fin:
            self.drug_molcule  = pickle.load(fin)

        # indexes = range(num_sample)
        # indexes = tqdm(indexes, "Constructing Dataset from pkl file: ")
        # get the index of protein and mol to quick build
        # for i in indexes:
            # protein = target_protein[protein_list[i]]
            # mol = drug_molcule[drug_list[i]]
            # md = coords_list[drug_list[i]]
            # self.data.append([protein, mol, md])
            # self.data.append([protein, mol])
        # ============================== Dataset Completing! ============================== #

    # TankBind split
    def deepdta_split(self):
        train_index = json.load(open(f"{self.path}train_index.txt"))
        valid_index = json.load(open(f"{self.path}valid_index.txt"))
        test_index = json.load(open(f"{self.path}test_index.txt"))  
        train_set = torch.utils.data.Subset(self, train_index) 
        valid_set = torch.utils.data.Subset(self, valid_index)
        test_set = torch.utils.data.Subset(self, test_index)
        return train_set, valid_set, test_set

    # PDBbind v2016 split
    def pdbbind2016_split(self, mode="general", fold=None):
        if mode == "general":
            train_index = json.load(open(f"{self.path}PDBbind_2016/split/train_index_general.txt"))
            test_index = json.load(open(f"{self.path}PDBbind_2016/split/test_index.txt"))
        elif mode == "refined":
            train_index = json.load(open(f"{self.path}PDBbind_2016/split/train_index_refined.txt"))
            test_index = json.load(open(f"{self.path}PDBbind_2016/split/test_index.txt"))
        elif mode == "ccv" and fold == 0:
            train_index = json.load(open(f"{self.path}PDBbind_2016/split/ccv_train0.txt"))
            test_index = json.load(open(f"{self.path}PDBbind_2016/split/ccv_test0.txt"))
        elif mode == "ccv" and fold == 1:
            train_index = json.load(open(f"{self.path}PDBbind_2016/split/ccv_train1.txt"))
            test_index = json.load(open(f"{self.path}PDBbind_2016/split/ccv_test1.txt"))
        elif mode == "ccv" and fold == 2:
            train_index = json.load(open(f"{self.path}PDBbind_2016/split/ccv_train2.txt"))
            test_index = json.load(open(f"{self.path}PDBbind_2016/split/ccv_test2.txt"))
        valid_set = []
        train_set = torch.utils.data.Subset(self, train_index)
        test_set = torch.utils.data.Subset(self, test_index)
        return train_set, valid_set, test_set

    # [11448, 2416, 4842] = 18706 LP-PDBbind split [11513, 2422, 4860] = 18795
    def ccv_split(self):
        dataset_file = self.path + 'pdbbind_datasets.csv'
        dataset_df = pd.read_csv(dataset_file)
        split_file = self.path + 'PDBbind_CCV_Split/LP_PDBBind.csv'
        split_df = pd.read_csv(split_file)
        split_df.rename(columns={"Unnamed: 0": "PDB_ID"}, inplace=True)
        train_index = dataset_df[dataset_df["PDB_ID"].isin(split_df[split_df["new_split"] == "train"]["PDB_ID"].str.upper())].index.tolist()
        valid_index = dataset_df[dataset_df["PDB_ID"].isin(split_df[split_df["new_split"] == "val"]["PDB_ID"].str.upper())].index.tolist()
        test_index = dataset_df[dataset_df["PDB_ID"].isin(split_df[split_df["new_split"] == "test"]["PDB_ID"].str.upper())].index.tolist()
        train_set = torch.utils.data.Subset(self, train_index)
        valid_set = torch.utils.data.Subset(self, valid_index)
        test_set = torch.utils.data.Subset(self, test_index)
        return train_set, valid_set, test_set

    # random split 4:1:1 as the num_samples
    def random_split(self):  
        train_set, valid_set, test_set = torch.utils.data.random_split(self, self.num_samples)
        return train_set, valid_set, test_set 

    def get_item(self, index):
        # load the protein from each file, drug in one file
        if self.lazy:
            file_path = '../../data/dta-datasets/PDBbind/ESM_Data/'
            protein_file = os.path.join(file_path, f'index_{index}.pkl')
            with open(protein_file, 'rb') as fin:
                graph1 = pickle.load(fin)
            graph2 = self.drug_molcule[index]
        else:
            graph1 = self.data[index][0]
            graph2 = self.data[index][1]
            # graph3 = self.data[index][2]
        if hasattr(graph1, "residue_feature"):
            with graph1.residue():
                graph1.residue_feature = graph1.residue_feature.to_dense()
                # add for label
                target = torch.as_tensor(self.label_list[index]["target"], dtype=torch.long)
                graph1.target = target
                mask = torch.as_tensor(self.label_list[index]["mask"], dtype=torch.bool)
                graph1.mask = mask

        # 将坐标的标签加入属性，就不会在运行时报错，因为batch的堆叠要求tensor维度得一致
        # with graph2.atom():
        #     graph2.coords = self.drug_coords[index] # trans the dim for batch pack

        item = {"graph1": graph1,
                "graph2": graph2,
                # "graph3": graph3
                }
        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item


@R.register("datasets.PDBbindProtein")
@utils.copy_args(data.ProteinDataset.load_pdbs)
class PDBbindProtein(data.ProteinDataset):
    """
    The 19350 proteins in PDBbind dataset for pocket label prediction. CCV split are used.
    Parameters:
        path (str): path to store the dataset
        protein_method (str): Protein loading method, from 'sequence','pdb'.
        transform (Callable, optional): protein sequence transformation function.
        lazy (bool, optional): if lazy mode is used, the protein-ligand pairs are processed in the dataloader.
            This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
        **kwargs
    """
    def __init__(
            self, path='../../data/dta-datasets/PDBbind/', protein_method='gearnet',
            transform=None, lazy=True, **kwargs
    ):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.transform = transform
        self.lazy = lazy
        # self.data = []

        # self.protein_file = self.path + 'pdbbind_proteins.csv'

        # ============================== Pkl file not exist, Creating one ==============================  #
        # protein_pkl = path + protein_method + '_Protein.pkl'
        # if not os.path.exists(protein_pkl):
        #     if protein_method == 'pdb' or 'gearnet':
        #         self.load_protein(self.protein_file, protein_method, **kwargs)
        #     else:
        #         raise ValueError("Protein method should be 'pdb', 'gearnet'!")

        # ============================== Loading Pkl file ==============================  #
        # with utils.smart_open(protein_pkl, "rb") as fin:
        #     target_protein = pickle.load(fin)

        # ============================== Loading label file ==============================  #
        label_pkl = path + 'gearnet_labeled.pkl'
        with utils.smart_open(label_pkl, "rb") as fin:
            self.label_list = pickle.load(fin)

        # get the index of protein and mol to quick build
        # for i in range(len(target_protein)):
        #     protein = target_protein[i]
        #     self.data.append(protein)
        # ============================== Dataset Completing! ============================== #

    def random_split(self):
        train_set, valid_set, test_set = torch.utils.data.random_split(self, [6450, 6450, 6450])
        return train_set, valid_set, test_set

    # [11448, 2416, 4842] = 18706 LP-PDBbind split [11513, 2422, 4860] = 18795
    def ccv_split(self):
        dataset_file = self.path + 'pdbbind_datasets.csv'
        dataset_df = pd.read_csv(dataset_file)
        split_file = self.path + 'PDBbind_CCV_Split/LP_PDBBind.csv'
        split_df = pd.read_csv(split_file)
        split_df.rename(columns={"Unnamed: 0": "PDB_ID"}, inplace=True)
        train_index = dataset_df[
            dataset_df["PDB_ID"].isin(split_df[split_df["new_split"] == "train"]["PDB_ID"].str.upper())].index.tolist()
        valid_index = dataset_df[
            dataset_df["PDB_ID"].isin(split_df[split_df["new_split"] == "val"]["PDB_ID"].str.upper())].index.tolist()
        test_index = dataset_df[
            dataset_df["PDB_ID"].isin(split_df[split_df["new_split"] == "test"]["PDB_ID"].str.upper())].index.tolist()
        train_set = torch.utils.data.Subset(self, train_index)
        valid_set = torch.utils.data.Subset(self, valid_index)
        test_set = torch.utils.data.Subset(self, test_index)
        return train_set, valid_set, test_set

    def get_item(self, index):
        # load the protein from each file, drug in one file
        if self.lazy:
            file_path = '../../data/dta-datasets/PDBbind/ESM_Data/'
            protein_file = os.path.join(file_path, f'index_{index}.pkl')
            with open(protein_file, 'rb') as fin:
                graph1 = pickle.load(fin)
        else:
            graph1 = self.data[index]
        if hasattr(graph1, "residue_feature"):
            with graph1.residue():
                graph1.residue_feature = graph1.residue_feature.to_dense()
                target = torch.as_tensor(self.label_list[index]["target"], dtype=torch.long)
                graph1.target = target
                mask = torch.as_tensor(self.label_list[index]["mask"], dtype=torch.bool)
                graph1.mask = mask
        item = {"graph1": graph1}

        if self.transform:
            item = self.transform(item)
        return item
