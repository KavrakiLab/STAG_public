from torch_geometric.explain import Explainer, CaptumExplainer
import numpy as np
import sys
import torch
import torch_geometric
from colour import Color

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


from models.STAG.encode_structure import get_graph, add_edge_data
from models.STAG.model import STAG
from models.STAG.utils import *

def get_grads(model,graph,target):
    model = model.to('cpu')
    model.eval()

    explainer = Explainer(
        model,
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config = dict(
            mode='binary_classification',
            task_level='graph',
            return_type='raw'
        ),
    )

    temp = torch_geometric.loader.DataLoader([graph],batch_size=1,shuffle=False)
    for sample in temp:
        explanation = explainer(target=torch.tensor(target),x=sample.x,edge_index=sample.edge_index,edge_attr=sample.edge_attr,batch=sample.batch)

    nm = explanation.node_mask
    nm = np.array(nm)
    norms = np.array([np.linalg.norm(nm[i,:]) for i in range(nm.shape[0])])
    return norms

def write_script(pdb, graph, grads, out_path, script_name):
    """
    ***Code adapted from:

    Ceder Dens, Wout Bittremieux, Fabio Affaticati, Kris Laukens, Pieter Meysman,
    Interpretable deep learning to uncover the molecular binding patterns determining TCR–epitope interaction predictions,
    ImmunoInformatics,
    Volume 11,
    2023,
    100027,
    ISSN 2667-1190,
    https://doi.org/10.1016/j.immuno.2023.100027.
    (https://www.sciencedirect.com/science/article/pii/S2667119023000071)
    Abstract: The recognition of an epitope by a T-cell receptor (TCR) is crucial for eliminating pathogens and establishing immunological memory. Prediction of the binding of any TCR–epitope pair is still a challenging task, especially for novel epitopes, because the underlying patterns are largely unknown to domain experts and machine learning models. To achieve a deeper understanding of TCR–epitope interactions, we have used interpretable deep learning techniques to gain insights into the performance of TCR–epitope binding machine learning models. We demonstrate how interpretable AI techniques can be linked to the three-dimensional structure of molecules to offer novel insights into the factors that determine TCR affinity on a molecular level. Additionally, our results show the importance of using interpretability techniques to verify the predictions of machine learning models for challenging molecular biology problems where small hard-to-detect problems can accumulate to inaccurate results.
    Keywords: T-cell epitope prediction; Interpretable deep learning; Immunoinformatics
    """
    is_p = np.array([1 if node.chain == cn_[0] else 0 for node in graph.nodes])
    p_aas = [node.res for node in graph.nodes if node.chain == cn_[0]]
    p_ids = [node.indx for node in graph.nodes if node.chain == cn_[0]]
    p_node_grads = grads[np.nonzero(is_p)]
    is_m = np.array([1 if node.chain == cn_[1] else 0 for node in graph.nodes])
    m_aas = [node.res for node in graph.nodes if node.chain == cn_[1]]
    m_ids = [node.indx for node in graph.nodes if node.chain == cn_[1]]
    m_node_grads = grads[np.nonzero(is_m)]
    is_a = np.array([1 if node.chain == cn_[2] else 0 for node in graph.nodes])
    a_aas = [node.res for node in graph.nodes if node.chain == cn_[2]]
    a_ids = [node.indx for node in graph.nodes if node.chain == cn_[2]]
    a_node_grads = grads[np.nonzero(is_a)]
    is_b = np.array([1 if node.chain == cn_[3] else 0 for node in graph.nodes])
    b_aas = [node.res for node in graph.nodes if node.chain == cn_[3]]
    b_ids = [node.indx for node in graph.nodes if node.chain == cn_[3]]
    b_node_grads = grads[np.nonzero(is_b)]

    # Color scale for highlighting
    color_scale = list(Color('lime').range_to(Color('red'), 101))  # Color scale from Green to Red, index from 0 to 100

    # Start writing the script
    script_file = open(f'{out_path}/{script_name}.pml', 'w')
    # Make sure PyMol loads the PDB file
    script_file.write(f'load {os.getcwd()}/{pdb}\n')
    # Set the background color white
    script_file.write('bg_color white\n')
    # Color all chians grey
    script_file.write(f'color grey80, chain {cn_[0]}\n')
    script_file.write(f'color grey80, chain {cn_[1]}\n')
    script_file.write(f'color grey80, chain {cn_[2]}\n')
    script_file.write(f'color grey80, chain {cn_[3]}\n')

    # Update colors of peptide
    for i in range(len(p_aas)):
        col = f'0x{color_scale[int(np.clip(p_node_grads[i],0,1) * 100)].get_hex_l()[1:]}'
        script_file.write(f'color {col}, chain {cn_[0]} and resi {p_ids[i]}\n')
    # Update colors of MHC
    for i in range(len(m_aas)):
        col = f'0x{color_scale[int(np.clip(m_node_grads[i],0,1) * 100)].get_hex_l()[1:]}'
        script_file.write(f'color {col}, chain {cn_[1]} and resi {m_ids[i]}\n')
    # Update colors of TCR_A
    for i in range(len(a_aas)):
        col = f'0x{color_scale[int(np.clip(a_node_grads[i],0,1) * 100)].get_hex_l()[1:]}'
        script_file.write(f'color {col}, chain {cn_[2]} and resi {a_ids[i]}\n')
    # Update colors of TCR_B
    for i in range(len(b_aas)):
        col = f'0x{color_scale[int(np.clip(b_node_grads[i],0,1) * 100)].get_hex_l()[1:]}'
        script_file.write(f'color {col}, chain {cn_[3]} and resi {b_ids[i]}\n')
    script_file.close()

if __name__ == '__main__':
    pdb_path = str(sys.argv[1])
    target = int(sys.argv[2])
    model_path = str(sys.argv[3])
    out_path = str(sys.argv[4])
    script_name = str(sys.argv[5])


    print('loading model')
    model = STAG()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.explaining=True

    print('loading graph')
    graph = get_graph(pdb_path,1,1)
    add_edge_data(graph)
    grads_graph = torch_geometric.data.Data(x=graph.get_node_data(),edge_index=graph.edge_index,edge_attr=graph.edge_attr,y=graph.label)

    print('calculating gradient with respect to target value 1')
    grads = get_grads(model,grads_graph,target)

    print('writing pymol visualization script')
    write_script(pdb_path, graph, grads, out_path, script_name)
    print('Complete! To visiualize results, just execute the pymol script: '+out_path+'/'+script_name+'.pml')
