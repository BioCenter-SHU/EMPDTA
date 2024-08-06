import os
import json
import random
from tqdm import tqdm
import pickle
import pandas as pd
from rdkit import Chem
from copy import deepcopy
from collections import defaultdict
import torch
from torch.utils import data as torch_data
from torchdrug import data, utils, core
from torchdrug.core import Registry as R

@R.register("datasets.Davis")
@utils.copy_args(data.ProteinLigandDataset.load_sequence)
class Davis(data.ProteinLigandDataset):
    """ 
    Input: 
        path = "../../data/dta-datasets/Davis/"
        method = 'sequence', 'pdb' or 'sdf' for different init instance methods.
    davis_datasets.csv ==> pd.series containing the all the col for usage!
    ["Drug"], ["Target"], ["Y"] and ["PDB_File"] ==> .tolist() for the input form list [].
    load_sequence/load_pdb/load_sdf function to build the dataset containing different attritubes.
    Notes: the class attributes do not need to be defined first. Once used they will be defined.
    - pdb_files: for protein 3d pdb file name and location
    - sequences: for protein 1d sequence
    - sdf: for drug 3d position
    - smiles: for drug 1d smiles
    - targets: for the affinity scores
    - data: for the torchdrug tuple(Protein, Molecule) for DTA task
    The interaction of 68(filtered protein under 10 interactions) kinase inhibitors with 442 kinases 
    covering>80% of the human catalytic protein kinome.(only 379 kinases are unique).
    Statistics for the 'whole' Dataset:
        - #Molecule: 68
        - #Protein: 442
        - #Interaction: 30,056
        - split:
        train:valid:test = 4:1:1 (and test shoule be hold out)
        - #Train: 20036
        - #Valid: 5010
        - #Test: 5010
    Parameters:
        path (str): path to store the dataset 
        drug_method (str): Drug loading method, from 'smile','2d' or '3d'. 
        protein_method (str): Protein loading method, from 'sequence','pdb'.
        description (str): whole(30056) or filter(9125) davis dataset
        transform (Callable, optional): protein sequence transformation function
        lazy (bool, optional): if lazy mode is used, the protein-ligand pairs are processed in the dataloader.
            This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
        **kwargs
    """
    splits = ["train", "valid", "test"]
    target_fields = ["affinity"]
    # Three init way, first the normal way using sequences,
    # another using the pdb/sdf for protein smiles for drug
    def __init__(self, path='../../data/dta-datasets/Davis/', drug_method='smile', protein_method='sequence',
                description='whole', transform=None, lazy=False, **kwargs):
        """
        - Use the DeepDTA davis dataset to create the local csv file.
        - Then init the Davis dataset instance by loading the local file to pd.DataFrame.
        - Select the col of the df to build the dataset object.
        """   
        # ============================== Path check and generate ============================== # 
        # os.path.expanduser expand the ~ to the full path
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.lazy = lazy
        self.transform = transform
        self.protein_method = protein_method  # added for get_item method
        self.description = description
        # ============================== Choose the local file for part/whole dataset ============================== #
        if self.description == 'filter':
            # 9,125 samples
            local_file = self.path + 'davis_filtered_datasets.csv'
            self.num_samples = [6085, 1520, 1520] 
        else:
            # 30,056 samples
            local_file = self.path + 'davis_datasets.csv'
            self.num_samples = [20036, 5010, 5010] 
        dataset_df = pd.read_csv(local_file)      
             
        # ============================== Get the col info of the Dataframe ============================== #
        drug_list = dataset_df["Drug_Index"].tolist() # for further index the drug in 68 durg list
        protein_list = dataset_df["protein_Index"].tolist()  # for further index the protein in 442 protein list
        self.pdb_files = dataset_df["PDB_File"].tolist()  # pdb file name like AF-Q2M2I8-F1-model_v4.pdb
        self.smiles = dataset_df["Drug"].tolist()
        self.targets = {}  # follow the torchdrug form create a dict to store
        self.targets["affinity"] = dataset_df["Y"].tolist()  # scaled to [0, 1]

        # ============================== Loading label file ==============================  #
        label_pkl = path + 'gearnet_labeled.pkl'
        with utils.smart_open(label_pkl, "rb") as fin:
            label_list = pickle.load(fin)

        # ============================== Generating the self.data list [protein, mol] ============================== #
        num_sample = len(self.smiles)
        for field, target_list in self.targets.items():
            if len(target_list) != num_sample:
                raise ValueError("Number of target `%s` doesn't match with number of molecules. "
                                 "Expect %d but found %d" % (field, num_sample, len(target_list)))

        self.data = []
        print(f'==================== Using {drug_method} drug method and {protein_method} protein method! ====================')

        # ============================== Loading Protein ============================== #
        protein_pkl = path + protein_method + '_Protein.pkl'
        # ============================== Pkl file not exist, Creating one ==============================  #
        if not os.path.exists(protein_pkl):
            protein_file = self.path + 'davis_proteins.csv'
            # 'gearnet' or 'comenet' are pre-defined through the graph construction
            if protein_method == 'pdb' or 'sequence' or 'gearnet' or 'comenet':
                self.load_protein(protein_file, protein_method, **kwargs)
            else:
                raise ValueError("Protein method should be 'pdb', 'sequence' 'gearnet' or 'comenet'!")
        # ============================== Loading Pkl file ==============================  #
        with utils.smart_open(protein_pkl, "rb") as fin:
            target_protein  = pickle.load(fin)
        # ============================== Loading Drug ============================== #
        drug_pkl = path + drug_method + '_Molecule.pkl'
        if not os.path.exists(drug_pkl):
            drug_file = self.path + 'davis_ligands.csv'
            if drug_method == 'smile' or '2d' or '3d' or 'comenet':
                self.load_drug(drug_file, drug_method, **kwargs)
            else:
                raise ValueError("Drug method should be 'smile' , '2d' '3d' or 'comenet' !")
        with utils.smart_open(drug_pkl, "rb") as fin:
            drug_molcule  = pickle.load(fin)

        # map the 442 protein label into 9125 DTA pairs
        self.label_list = []
        indexes = range(num_sample)
        indexes = tqdm(indexes, "Constructing Dataset from pkl file: ")
        # get the index of protein and mol to quick build
        for i in indexes:
            protein = target_protein[protein_list[i]]
            mol = drug_molcule[drug_list[i]]
            self.label_list.append(label_list[protein_list[i]])
            self.data.append([protein, mol])
        # ============================== Dataset Completing! ============================== #

    # sequence split
    def split(self, keys=None):  
        keys = keys or self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        return splits

    # random split 4:1:1 as the num_samples
    def random_split(self):  
        train_set, valid_set, test_set = torch.utils.data.random_split(self, self.num_samples)
        return train_set, valid_set, test_set 

    # fixed index as the DeepDTA provided, we need to reproduce the 5 folds for training, 1 fold for test
    def deepdta_split(self, fold):
        if self.description == 'filter':
            train_valid_index = json.load(open("../../data/dta-datasets/Davis/train_fold_setting_for_filter_davis.txt"))
            test_index = json.load(open("../../data/dta-datasets/Davis/test_fold_setting_for_filter_davis.txt"))   
        else:        
            train_valid_index = json.load(open("../../data/dta-datasets/Davis/train_fold_setting1.txt"))
            test_index = json.load(open("../../data/dta-datasets/Davis/test_fold_setting1.txt"))
        test_set = torch.utils.data.Subset(self, test_index)
        print(f'==================== Training on Fold: {fold} ====================')
        train_index = []
        valid_index = []
        valid_index = train_valid_index[fold]  # which is the valid list for dataset build
        otherfolds = deepcopy(train_valid_index)  # copy a new list without effecting the raw list
        otherfolds.pop(fold)  # pop out the valid fold and left the other 4 folds for training
        for train_fold in otherfolds:  # to merge the 4 fold into a single train_index list
            train_index.extend(train_fold)
        # Get the list, now build the dataset
        valid_set = torch.utils.data.Subset(self, valid_index) 
        train_set = torch.utils.data.Subset(self, train_index)
        return train_set, valid_set, test_set

    # Clustered cross-validation 3-Fold
    # from 'Latent Biases in Machine Learning Models for Predicting Binding Affinities Using Popular Data Sets'
    def ccv_split(self, fold, mode="target"):
        if self.description == 'filter' and mode == "target":
            fold0_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/FilteredDavis_target_ccv0.csv"
            fold1_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/FilteredDavis_target_ccv1.csv"
            fold2_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/FilteredDavis_target_ccv2.csv"
        elif self.description == 'whole' and mode == "target":
            fold0_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/Davis_target_ccv0.csv"
            fold1_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/Davis_target_ccv1.csv"
            fold2_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/Davis_target_ccv2.csv"
        fold0_index = pd.read_csv(fold0_file, header=None)
        fold0_index = fold0_index[0].to_numpy().tolist()
        fold1_index = pd.read_csv(fold1_file, header=None)
        fold1_index = fold1_index[0].to_numpy().tolist()
        fold2_index = pd.read_csv(fold2_file, header=None)
        fold2_index = fold2_index[0].to_numpy().tolist()
        if fold == 0:
            train_index = []
            train_index.extend(fold1_index)
            train_index.extend(fold2_index)
            train_set = torch.utils.data.Subset(self, train_index)
            test_set = torch.utils.data.Subset(self, fold0_index)
        elif fold == 1:
            train_index = []
            train_index.extend(fold0_index)
            train_index.extend(fold2_index)
            train_set = torch.utils.data.Subset(self, train_index)
            test_set = torch.utils.data.Subset(self, fold1_index)
        else:
            train_index = []
            train_index.extend(fold0_index)
            train_index.extend(fold1_index)
            train_set = torch.utils.data.Subset(self, train_index)
            test_set = torch.utils.data.Subset(self, fold2_index)
        valid_set = []  # empty
        print(f'==================== Training on Mode {mode} Fold-{fold} ====================')
        return train_set, valid_set, test_set

    def get_item(self, index):
        if self.lazy:
            graph1 = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
            mol = Chem.MolFromSmiles(self.smiles[index])
            if not mol:
                graph2 = None
            else:
                graph2 = data.Molecule.from_molecule(mol, **self.kwargs)
        else:
            graph1 = self.data[index][0]
            graph2 = self.data[index][1]
        if hasattr(graph1, "residue_feature"):
            with graph1.residue():
                graph1.residue_feature = graph1.residue_feature.to_dense()
                target = torch.as_tensor(self.label_list[index]["target"], dtype=torch.long)
                graph1.target = target
                mask = torch.as_tensor(self.label_list[index]["mask"], dtype=torch.bool)
                graph1.mask = mask

        item = ({
            "graph1": graph1,
            "graph2": graph2,
        })

        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item


@R.register("datasets.FilteredDavis")
@utils.copy_args(data.ProteinLigandDataset.load_sequence)
class FilteredDavis(data.ProteinLigandDataset):
    """
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

    def __init__(self, path='../../data/dta-datasets/Davis/', drug_method='smile', protein_method='sequence',
                 transform=None, lazy=False, **kwargs):
        # ============================== Path check and generate ============================== # 
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.lazy = lazy
        self.transform = transform
        self.protein_method = protein_method  # added for get_item method
        # ============================== Choose the local file for part/whole dataset ============================== #
        # 9,125 samples
        local_file = self.path + 'davis_filtered_datasets.csv'
        self.num_samples = [6085, 1520, 1520] 
        dataset_df = pd.read_csv(local_file)      
             
        # ============================== Get the col info of the Dataframe ============================== #
        drug_list = dataset_df["Drug_Index"].tolist() # for further index the drug in 68 durg list
        protein_list = dataset_df["protein_Index"].tolist()  # for further index the protein in 442 protein list
        self.pdb_files = dataset_df["PDB_File"].tolist()  # pdb file name like AF-Q2M2I8-F1-model_v4.pdb
        self.smiles = dataset_df["Drug"].tolist()
        self.targets = {}  # follow the torchdrug form create a dict to store defaultdict(list)
        self.targets["affinity"] = dataset_df["Y"].tolist()  # scaled to [0, 1]

        # ============================== Loading drug 3d file ==============================  #
        # coords_pkl = path + '3d_Molecule.pkl' # for distance
        # with utils.smart_open(coords_pkl, "rb") as fin:
        #       coords_list = pickle.load(fin)

        # ============================== Loading label file ==============================  #
        label_pkl = path + 'gearnet_labeled.pkl'
        with utils.smart_open(label_pkl, "rb") as fin:
            label_list = pickle.load(fin)

        # ============================== Generating the self.data list [protein, mol] ============================== #
        num_sample = len(self.smiles)                
        for field, target_list in self.targets.items():
            if len(target_list) != num_sample:
                raise ValueError("Number of target `%s` doesn't match with number of molecules. "
                                 "Expect %d but found %d" % (field, num_sample, len(target_list)))

        self.data = []
        print(f'==================== Using {drug_method} drug method and {protein_method} protein method! ====================')

        # ============================== Loading Protein ============================== #
        protein_pkl = path + protein_method + '_Protein.pkl'
        # ============================== Pkl file not exist, Creating one ==============================  #
        if not os.path.exists(protein_pkl):
            protein_file = self.path + 'davis_proteins.csv'
            # 'gearnet' or 'comenet' are pre-defined through the graph construction
            if protein_method == 'pdb' or 'sequence' or 'gearnet' or 'comenet':
                self.load_protein(protein_file, protein_method, **kwargs)
            else:
                raise ValueError("Protein method should be 'pdb', 'sequence' 'gearnet' or 'comenet'!")
        # ============================== Loading Pkl file ==============================  #
        with utils.smart_open(protein_pkl, "rb") as fin:
            target_protein = pickle.load(fin)

        # ============================== Loading Drug ============================== #
        drug_pkl = path + drug_method + '_Molecule.pkl'
        # ============================== Pkl file not exist, Creating one ==============================  #
        if not os.path.exists(drug_pkl):
            drug_file = self.path + 'davis_ligands.csv'
            if drug_method == 'smile' or '2d' or '3d' or 'comenet':
                self.load_drug(drug_file, drug_method, **kwargs)
            else:
                raise ValueError("Drug method should be 'smile' , '2d' '3d' or 'comenet' !")
        # ============================== Loading Pkl file ==============================  #
        with utils.smart_open(drug_pkl, "rb") as fin:
            drug_molcule  = pickle.load(fin)

        # map the 442 protein label into 9125 DTA pairs
        self.label_list = []
        indexes = range(num_sample)
        indexes = tqdm(indexes, "Constructing Dataset from pkl file: ")
        # get the index of protein and mol to quick build
        for i in indexes:
            protein = target_protein[protein_list[i]]
            drug = drug_molcule[drug_list[i]]
            self.label_list.append(label_list[protein_list[i]])
            self.data.append([protein, drug])

        # ============================== Dataset Completing! ============================== #

    # sequence split
    def split(self, keys=None):  
        keys = keys or self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        return splits

    # random split 4:1:1 as the num_samples
    def random_split(self):  
        train_set, valid_set, test_set = torch.utils.data.random_split(self, self.num_samples)
        return train_set, valid_set, test_set 

    def deepdta_split(self, fold):
        # fixed index as the DeepDTA provided, we need to reproduce the 5 folds for training, 1 fold for test
        train_valid_index = json.load(open("../../data/dta-datasets/Davis/train_fold_setting_for_filter_davis.txt"))
        test_index = json.load(open("../../data/dta-datasets/Davis/test_fold_setting_for_filter_davis.txt"))
        test_set = torch.utils.data.Subset(self, test_index)
        print(f'==================== Training on Fold: {fold} ====================')
        train_index = []
        valid_index = []
        valid_index = train_valid_index[fold]  # which is the valid list for dataset build
        otherfolds = deepcopy(train_valid_index)  # copy a new list without effecting the raw list
        otherfolds.pop(fold)  # pop out the valid fold and left the other 4 folds for training
        for train_fold in otherfolds:  # to merge the 4 fold into a single train_index list
            train_index.extend(train_fold)
        # Get the list, now build the dataset
        valid_set = torch.utils.data.Subset(self, valid_index) 
        train_set = torch.utils.data.Subset(self, train_index)
        return train_set, valid_set, test_set

    # Clustered cross-validation 3-Fold
    # from 'Latent Biases in Machine Learning Models for Predicting Binding Affinities Using Popular Data Sets'
    def ccv_split(self, fold):
        fold0_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/FilteredDavis_target_ccv0.csv"
        fold1_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/FilteredDavis_target_ccv1.csv"
        fold2_file = "../../data/dta-datasets/Davis/Davis_CCV_Split/FilteredDavis_target_ccv2.csv"
        fold0_index = pd.read_csv(fold0_file, header=None)
        fold0_index = fold0_index[0].to_numpy().tolist()
        fold1_index = pd.read_csv(fold1_file, header=None)
        fold1_index = fold1_index[0].to_numpy().tolist()
        fold2_index = pd.read_csv(fold2_file, header=None)
        fold2_index = fold2_index[0].to_numpy().tolist()
        if fold == 0:
            train_index = []
            train_index.extend(fold1_index)
            train_index.extend(fold2_index)
            train_set = torch.utils.data.Subset(self, train_index)
            test_set = torch.utils.data.Subset(self, fold0_index)
        elif fold == 1:
            train_index = []
            train_index.extend(fold0_index)
            train_index.extend(fold2_index)
            train_set = torch.utils.data.Subset(self, train_index)
            test_set = torch.utils.data.Subset(self, fold1_index)
        else:
            train_index = []
            train_index.extend(fold0_index)
            train_index.extend(fold1_index)
            train_set = torch.utils.data.Subset(self, train_index)
            test_set = torch.utils.data.Subset(self, fold2_index)
        valid_set = []  # empty
        print(f'==================== Training on CCV Fold-{fold} ====================')
        return train_set, valid_set, test_set


    def get_item(self, index):
        if self.lazy:
            graph1 = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
            mol = Chem.MolFromSmiles(self.smiles[index])
            if not mol:
                graph2 = None
            else:
                graph2 = data.Molecule.from_molecule(mol, **self.kwargs)
        else:
            graph1 = self.data[index][0]
            graph2 = self.data[index][1]
        if hasattr(graph1, "residue_feature"):
            with graph1.residue():
                graph1.residue_feature = graph1.residue_feature.to_dense()
                target = torch.as_tensor(self.label_list[index]["target"], dtype=torch.long)
                graph1.target = target
                mask = torch.as_tensor(self.label_list[index]["mask"], dtype=torch.bool)
                graph1.mask = mask

        item = ({
            "graph1": graph1,
            "graph2": graph2,
        })

        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item


@R.register("datasets.DavisProtein")
@utils.copy_args(data.ProteinDataset.load_sequence)
class DavisProtein(data.ProteinDataset):
    """
    The 379 proteins in Davis dataset for pocket label prediction. CCV split are used.
    Parameters:
        path (str): path to store the dataset
        protein_method (str): Protein loading method, from 'sequence','pdb'.
        transform (Callable, optional): protein sequence transformation function.
        **kwargs
    """
    def __init__(self, path='../../data/dta-datasets/Davis/', protein_method='sequence', transform=None, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.transform = transform
        self.data = []

        self.protein_file = self.path + 'davis_proteins.csv'

        # ============================== Pkl file not exist, Creating one ==============================  #
        protein_pkl = path + protein_method + '_Protein.pkl'
        if not os.path.exists(protein_pkl):
            if protein_method == 'pdb' or 'sequence' or 'gearnet' or 'comenet':
                self.load_protein(self.protein_file, protein_method, **kwargs)
            else:
                raise ValueError("Protein method should be 'pdb', 'sequence' 'gearnet' or 'comenet'!")

        # ============================== Loading Pkl file ==============================  #
        with utils.smart_open(protein_pkl, "rb") as fin:
            target_protein = pickle.load(fin)

        # ============================== Loading label file ==============================  #
        label_pkl = path + 'gearnet_labeled.pkl'  # gearnet_labeled pocket_uniprot_label
        with utils.smart_open(label_pkl, "rb") as fin:
            self.label_list = pickle.load(fin)

        # get the index of protein and mol to quick build
        for i in range(len(target_protein)):
            protein = target_protein[i]
            self.data.append(protein)
        # ============================== Dataset Completing! ============================== #

    def random_split(self):
        train_set, valid_set, test_set = torch.utils.data.random_split(self, [147, 147, 148])
        return train_set, valid_set, test_set

    # Clustered cross-validation 3-Fold
    # from 'Latent Biases in Machine Learning Models for Predicting Binding Affinities Using Popular Data Sets'
    def ccv_split(self, fold=0):
        json_file = self.path + "/Davis_CCV_Split/3_folds_target.json"
        fold_dict = json.load(open(json_file))
        protein_df = pd.read_csv(self.protein_file)

        fold0_index = []
        fold1_index = []
        fold2_index = []
        # mapping the name into index of CCV three folds
        for index in range(len(protein_df['Gene'])):
            gene_name = protein_df['Gene'][index]
            if gene_name in fold_dict['fold0']:
                fold0_index.append(index)
            elif gene_name in fold_dict['fold1']:
                fold1_index.append(index)
            elif gene_name in fold_dict['fold2']:
                fold2_index.append(index)

        if fold == 0:
            train_set = torch.utils.data.Subset(self, fold1_index)
            valid_set = torch.utils.data.Subset(self, fold2_index)
            test_set = torch.utils.data.Subset(self, fold0_index)
        elif fold == 1:
            train_set = torch.utils.data.Subset(self, fold0_index)
            valid_set = torch.utils.data.Subset(self, fold2_index)
            test_set = torch.utils.data.Subset(self, fold1_index)
        else:
            train_set = torch.utils.data.Subset(self, fold1_index)
            valid_set = torch.utils.data.Subset(self, fold0_index)
            test_set = torch.utils.data.Subset(self, fold2_index)
        print(f'==================== Training on Fold-{fold} ====================')
        return train_set, valid_set, test_set

    def get_item(self, index):
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

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))
