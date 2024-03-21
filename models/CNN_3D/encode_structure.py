from Bio.PDB import PDBParser

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.transform import Rotation

import warnings
warnings.filterwarnings('ignore')

from .utils import *

def align_structure(structure):
    pep_CAs = []
    MHC_CAs = []
    TCR_CAs = []
    for chain in structure[0]:
        if cn[chain.id] == 0:
            for res in chain:
                for atom in res.get_atoms():
                    if atom.get_name() == 'CA':
                        pep_CAs.append(atom.get_coord())
        elif cn[chain.id] == 1:
            for res in chain:
                for atom in res.get_atoms():
                    if atom.get_name() == 'CA':
                        MHC_CAs.append(atom.get_coord())
        else:
            for res in chain:
                for atom in res.get_atoms():
                    if atom.get_name() == 'CA':
                        TCR_CAs.append(atom.get_coord())
    pep_CAs = np.array(pep_CAs)
    translation = np.mean(pep_CAs,axis=0)
    pep_CAs = pep_CAs - translation
    MHC_CAs = np.array(MHC_CAs) - translation
    TCR_CAs = np.array(TCR_CAs) - translation
    def pep_axis_dis(theta):
        rot_mat = Rotation.from_euler("XYZ",theta)
        rot_pep_CAs = (rot_mat.as_matrix()@pep_CAs.T)
        return np.sum((rot_pep_CAs[1,:]**2 + rot_pep_CAs[2,:]**2)**0.5)
    res1 = minimize(pep_axis_dis, np.array([0,0,0]))
    MHC_CAs = (Rotation.from_euler("XYZ",res1.x).as_matrix()@MHC_CAs.T)
    TCR_CAs = (Rotation.from_euler("XYZ",res1.x).as_matrix()@TCR_CAs.T)
    def rot_about_axis(theta):
        rot_mat = Rotation.from_euler("XYZ",[theta,0,0])
        rot_mhc_CAs = (rot_mat.as_matrix()@MHC_CAs)
        rot_tcr_CAs = (rot_mat.as_matrix()@TCR_CAs)
        return np.sum(np.reciprocal(np.exp(-2*rot_mhc_CAs[1,:])+1)-1) -  np.sum(2*np.reciprocal(np.exp(-2*rot_tcr_CAs[1,:])+1)-1)

    res2 = minimize_scalar(rot_about_axis)
    return translation,Rotation.from_euler("XYZ",[res2.x,0,0]).as_matrix()@Rotation.from_euler("zyx",res1.x).as_matrix()

def voxelize(pdb):
    parser = PDBParser()
    structure = parser.get_structure('roi',pdb)
    trans, rot = align_structure(structure)
    vox = np.zeros((24,36,36,36))

    for chain in structure[0]:
        for res in chain:
            for atom in res.get_atoms():
                if atom.get_name() == 'CA':
                    c = (atom.get_coord()-trans)@rot
                    r = radii[res.get_resname()]

                    for x in range(max(-18,int(np.trunc(c[0]-r))),min(18,int(np.trunc(c[0]+r)))):
                        for y in range(max(-18,int(np.trunc(c[1]-r))),min(18,int(np.trunc(c[1]+r)))):
                            for z in range(max(-18,int(np.trunc(c[2]-r))),min(18,int(np.trunc(c[2]+r)))):
                                vox[AA_1hot_idx[res.get_resname()],x+18,y+18,z+18] = 1
                                vox[cn[chain.id]+20,x+18,y+18,z+18] = 1

    return vox
