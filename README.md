# EMPDTA: An End-to-End Multimodal Representation Learning Framework with Pocket Online Detection for Drug-Target Affinity Prediction 
The official code for [EMPDTA](https://doi.org/10.3390/molecules29122912).

This framework introduces an online generation method for protein surfaces based on residue-scale point cloud sampling and utilizes multi-layer quasi-geodesic convolution to aggregate local features of point clouds, creating "fingerprint" of protein structure surfaces. A protein surface encoding module is designed for the online prediction of protein pockets. EMPDTA further integrates this module into the prediction framework, providing interaction regions to narrow the computational scope before affinity prediction. The visualization results demonstrate that the framework, through an end-to-end approach, achieves single-input multi-task output, allowing simultaneous prediction of protein pockets and drug-target affinity.

![Figure](https://github.com/BioCenter-SHU/EMPDTA/blob/main/figures/EMPDTA.png)

# Installation

## Data

All the raw data came from the previous work. The sequence-based datasets (Filtered Davis, Davis and KIBA) and the split came from [DeepDTA](https://github.com/hkmztrk/DeepDTA) and [MDeePred](https://github.com/cansyl/MDeePred/). The structure-based dataset PDBbind and the split could be download from the Official [website](http://pdbbind.org.cn/index.php).

We customized the dataset construction file following [torchdrug](https://torchdrug.ai/) format which torchdrug did not provide (`torchdrug/datasets/davis.py`). To use our dataset file directly like, we suggest you just replace our `torchdrug`  file with you installation package(env file in you conda).  

As for pocket labels, the above sequence-based datasets (Davis, Filtered Davis, and KIBA) miss the information about the binding pockets. Therefore, we collect the pocket labels of the corresponding protein from the UniProt website (https://www.uniprot.org/). However, the structure-based PDBbind dataset conveniently offers structural files for protein pockets, facilitating their use as labels through straightforward index correspondence.

## Requirements

You'll need to follow the TorchDrug [installation](https://torchdrug.ai/docs/installation.html) which is our main platform. The conda environment have been tested in Linux/Windows. The detailed requirement can be found in `requirements.txt`.
The main dependencies are listed below:
*  scikit-learn=1.1.3
*  torch=1.13.1+cu117
*  torchdrug=0.2.0.post1
*  torch-geometric=2.2.0
*  wandb=0.14.0

# Usage

## Dataset construction

Scripts for all four dataset construction are provided in [PocketDTA](https://github.com/BioCenter-SHU/PocketDTA/tree/main/script/notebook/DatasetBuilding) in the `script/notebook/DatasetBuilding/` file. The notebook file would generate protein and drug pkl file.

As for pocket labels, the three below notebook will generate the label pkl file.

* `DavisPocketAnnotation.ipynb`
* `KIBAPocketAnnotation.ipynb`
* `PDBbindPocketAnnotation.ipynb`

The file `torchdrug/datasets/davis.py` in torchdrug then provide the whole picture for Davis and Filtered Davis dataset.

## Prediction

The single sun and wandb sweep file can be found in `script/pythonfile/`, after enter your conda env just use `python MTL_FD_AIO_Sweep.py` could sweep on Filterd Davis dataset.


