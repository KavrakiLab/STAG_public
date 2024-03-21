from Bio.PDB import PDBParser
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from .utils import *

def is_contact(res1,res2,radius=6):
    for atom2 in res2.get_atoms():
        # if atom2.get_name() in ['CA','CB','C','N','O']:
        if atom2.get_name() in ['CA']:
            for atom in res1.get_atoms():
                # if atom.get_name() in ['CA','CB','C','N','O']:
                if atom.get_name() in ['CA']:
                    if np.linalg.norm(atom.get_coord()-atom2.get_coord()) < radius:
                        return 1
    return 0

def get_contacts(pdb):
    parser = PDBParser()
    structure = parser.get_structure('tcr_pmch',pdb)
    mat = np.zeros((20,20))
    for chain in structure[0]:
        if cn[chain.id] in [0,1]:
            for res1 in chain:
                for chain2 in structure[0]:
                    if cn[chain2.id] in [2,3]:
                        for res2 in chain2:
                            mat[dn[d[res1.get_resname()]],dn[d[res2.get_resname()]]] += is_contact(res1,res2)
    return mat.flatten()
