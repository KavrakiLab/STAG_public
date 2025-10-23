<img align="right" src="https://github.com/KavrakiLab/STAG_public/blob/main/images/STAG_logo.png" width=25% height=25%>

# STAG

## **S**tructural **T**CR **A**nd pMHC binding specificity prediction **G**raph neural network

## Usage

### Evaluations
The `evaluations.py` script is used for performing cross validation of the different structure-based classifiers on the datasets. It takes the following arguments:
- `method`
  - the method to evaluate
  - 'STAG', 'CNN_3D', or 'Contacts'
- `dataset`
  - on which dataset to perform the cross validation
  - 'GLCTLVAML', 'IVTDFSVIK', 'NLVPMVATV', 'ELAGIGILTV', 'RAKFKQLL', 'GILGFVFTL', 'AVFDRKSDAK', 'KLGGALQAK', or 'pan_peptide'

The script will perform 5-fold cross validation of the given method and save the trained version of the model in the outputs file. These trained models may then be used to make and interpret predictions for new TCR-pMHC complexes.

Example:
`python evaluations.py STAG ELAGIGILTV`

### Interpretations
The `interpretability.py` script is used for creating pymol files that show visual explanations of the predictions given by a trained STAG model. It takes the following arguments:
- `pdb_path`
  - the path to the pdb on which to make and visualize a prediciton
- `target`
  - the target prediction (0 or 1) to be used in the integrated gradients algorithm
- `model_path`
  - the path to the trained STAG model `.pt` file to use when making predictions
- `out_path`
  - the path to save the generated pymol script
- `script_name`
  - the name to give the generated pymol script

Example pymol scripts are included in the `pymol_scripts` directory

Example:
`python interpretability.py datasets/pan_peptide/structures/365.pdb 1 output/STAG_sample_model.pt pymol_scripts pan_pep_pdb365_sample_model_grad_viz`

## Project Organization 
```bash
STAG_public
├───datasets
│   ├───AVFDRKSDAK
│   │   └───structures
│   ├───ELAGIGILTV
│   │   └───structures
│   ├───GILGFVFTL
│   │   └───structures
│   ├───GLCTLVAML
│   │   └───structures
│   ├───IVTDFSVIK
│   │   └───structures
│   ├───KLGGALQAK
│   │   └───structures
│   ├───NLVPMVATV
│   │   └───structures
│   ├───pan_peptide
│   │   └───structures
│   └───RAKFKQLL
│       └───structures
├───models
│   ├───CNN_3D
│   ├───Contacts
│   └───STAG
├───output
└───pymol_scripts
```

### Models
The models directories each contain the following files:
```bash
model.py                  - python file to construct the machein learning model
encode_structure.py       - python file to parse pdbs and convert them to the structure representation used by the model
cross_validation.py       - python file to perform cross validaiton of the model
utils.py                  - python file containg utils funcitons and variables used by the model
```

### Data
The datasts directories contain a csv detailing the attributes of each TCR-pMHC pair in the dataset and a `structures` sub-directory containng the modeled structures of each TRC-pMHC pair in the dataset. 
All structues are saved as pdb files of the form `###.pdb` where `###` is the number cooresponding to that structure's index in the csv file. The chains in each PDB file are labeled **P** : *peptide*, **M** : *MHC*, **A** : *TCR α-chian*, **B** : TCR *β-chain*

## Citation
J. K. Slone, A. Conev, M. M. Rigo, A. Reuben and L. E. Kavraki, "TCR-pMHC Binding Specificity Prediction From Structure Using Graph Neural Networks," in IEEE Transactions on Computational Biology and Bioinformatics, vol. 22, no. 1, pp. 171-179, Jan.-Feb. 2025, doi: https://doi.org/10.1109/TCBBIO.2024.3504235
