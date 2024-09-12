import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import dotenv
dotenv.load_dotenv(".env")
import matplotlib.pyplot as plt
from matplotlib import colormaps

import os
import argparse
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
# import torch_cluster

import biotite
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.analysis.rms import rmsd as get_rmsd

from src.data.data_utils import pdb_to_tensor, pdb_to_tensor_amir, get_twist_amir, get_c4p_coords, get_backbone_coords, dist_amir
from src.data.clustering_utils import cluster_sequence_identity, cluster_structure_similarity
from src.data.sec_struct_utils import pdb_to_x3dna_amir, x3dna_to_sec_struct_amir,get_unpaird_amir
from src.data.featurizer import normed_vec, normed_cross, get_angle_amir, spheric2cartesian

import warnings
warnings.filterwarnings("ignore", category=biotite.structure.error.IncompleteStructureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

DATA_PATH = os.environ.get("DATA_PATH")

keep_insertions = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', dest='project_name', default='gRNAde_v2', type=str)
    parser.add_argument('--entity', dest='entity', default='amanzour', type=str)
    parser.add_argument('--expt_name', dest='expt_name', default='process_data', type=str)
    parser.add_argument('--tags', nargs='+', dest='tags', default=[])
    parser.add_argument('--no_wandb', action="store_true")
    args, unknown = parser.parse_known_args()

    # Initialise wandb
    #if args.no_wandb:
    wandb.init(project=args.project_name, entity=args.entity, name=args.expt_name, mode='disabled')
    #else:
    #    wandb.init(
    #        project=args.project_name, 
    #        entity=args.entity,
    #        name=args.expt_name, 
    #        tags=args.tags,
    #        mode='online'
    #    )

    print("\nLoading non-redundant equivalence class table")
    eq_class_table = pd.read_csv(os.path.join(DATA_PATH, "nrlist_3.306_4.0A.csv"), names=["eq_class", "representative", "members"], dtype=str)
    eq_class_table.eq_class = eq_class_table.eq_class.apply(lambda x: x.split("_")[2].split(".")[0])

    id_to_eq_class = {}
    eq_class_to_ids = {}
    for i, row in tqdm(eq_class_table.iterrows(), total=len(eq_class_table)):
        ids_in_class = []
        for member in row["members"].split(","):
            _member = member.replace("|", "_")
            _chains = _member.split("+")
            if len(_chains) > 1:
                _member = _chains[0]
                for chain in _chains[1:]:
                    _member += f"-{chain.split('_')[2]}"

            id_to_eq_class[_member] = row["eq_class"]
            ids_in_class.append(_member)
        
        eq_class_to_ids[row["eq_class"]] = ids_in_class

    print("\nLoading RNAsolo table")
    rnasolo_table = pd.read_csv(os.path.join(DATA_PATH, "rnasolo-main-table.csv"), dtype=str)
    rnasolo_table.eq_class = rnasolo_table.eq_class.apply(lambda x: str(x).split(".")[0])

    eq_class_to_type = {}
    for i, row in tqdm(rnasolo_table.iterrows(), total=len(rnasolo_table)):
        eq_class_to_type[row["eq_class"]] = row["molecule"]

    print("\nLoading RFAM table")
    rfam_table = pd.read_csv(os.path.join(DATA_PATH, "RFAM_families_27062023.csv"), dtype=str)

    id_to_rfam = {}
    for i, row in tqdm(rfam_table.iterrows(), total=len(rfam_table)):
        if row["pdb_id"].upper() not in id_to_rfam.keys():
            id_to_rfam[row["pdb_id"].upper()] = row["id"]

    # Initialise empty dictionaries
    id_to_seq = {}
    seq_to_data = {}
    error_ids = []

    print(f"\nProcessing raw PDB files from {DATA_PATH}")
    filenames = tqdm(os.listdir(os.path.join(DATA_PATH, "raw")))


    for filename in filenames:
        try:
            structure_id, file_ext = os.path.splitext(filename)
            filenames.set_description(structure_id)
            if file_ext != ".pdb": continue

            sequence, coords, sec_struct, sasa = pdb_to_tensor_amir(
                os.path.join(DATA_PATH, "raw", filename),
                keep_insertions=keep_insertions,
                keep_pseudoknots=False
            )

            # basic post processing validation:
            # do not include sequences with less than 10 nucleotides,
            # which is the minimum length for sequence identity clustering
            if len(sequence) <= 50: 
                continue

            # get RFAM family
            rfam = id_to_rfam[structure_id.split("_")[0]] if \
                structure_id.split("_")[0] in id_to_rfam.keys() else "unknown"

            # get non-redundant equivalence class
            eq_class = id_to_eq_class[structure_id] if structure_id in \
                id_to_eq_class.keys() else "unknown"

            # get structure type (solo RNA, RNA-protein, RNA-DNA)
            struct_type = eq_class_to_type[eq_class] if eq_class in \
                eq_class_to_type.keys() else "unknown"

            # update dictionary    
            if sequence in seq_to_data.keys():
                # align coords of current structure to first entry
                coords_0 = seq_to_data[sequence]['coords_list'][0]
                R_hat = rotation_matrix(
                    get_c4p_coords(coords),  # mobile set
                    get_c4p_coords(coords_0) # reference set
                )[0]
                coords = coords @ R_hat.T

                # compute C4' RMSD of current structure to all other structures
                for other_id, other_coords in zip(seq_to_data[sequence]['id_list'], seq_to_data[sequence]['coords_list']):
                    seq_to_data[sequence]['rmsds_list'][(structure_id, other_id)] = get_rmsd(
                        get_c4p_coords(coords), 
                        get_c4p_coords(other_coords), 
                        superposition=True
                    )

                seq_to_data[sequence]['id_list'].append(structure_id)
                seq_to_data[sequence]['coords_list'].append(coords.float())
                seq_to_data[sequence]['sec_struct_list'].append(sec_struct)
                seq_to_data[sequence]['sasa_list'].append(sasa)
                seq_to_data[sequence]['rfam_list'].append(rfam)
                seq_to_data[sequence]['eq_class_list'].append(eq_class)
                seq_to_data[sequence]['type_list'].append(struct_type)
            
            # create new entry for new sequence
            else:
                seq_to_data[sequence] = {
                    'sequence': sequence,               # sequence string
                    'k': [structure_id],          # list of PDB IDs
                    'coords_list': [coords.float()],    # list of 3D coordinates of shape ``(length, 27, 3)``
                    'sec_struct_list': [sec_struct],    # list of secondary structure base pairs tuples
                    'sasa_list': [sasa],                # list of SASA values of shape ``(length, )``
                    'rfam_list': [rfam],                # list of RFAM family IDs
                    'eq_class_list': [eq_class],        # list of non-redundant equivalence class IDs
                    'type_list': [struct_type],         # list of structure types
                    'rmsds_list': {},                   # dictionary of pairwise C4' RMSD values between structures
                    'cluster_seqid0.8': -1,             # cluster ID of sequence identity clustering at 80%
                    'cluster_structsim0.45': -1         # cluster ID of structure similarity clustering at 45%
                }

            id_to_seq[structure_id] = sequence
        
        # catch errors and check manually later
        except Exception as e:
            print(structure_id, e)
            error_ids.append((structure_id, e))


    twists = []
    nts ={'A':0,'C':0,'G':0,'U':0}
    nts_unpaired ={'A':0,'C':0,'G':0,'U':0}

    for key_amir,value_amir in seq_to_data.items():

        print(value_amir['sequence'])
        print(value_amir['k'])
        cur_sec_struct = value_amir['sec_struct_list'][0]
        cur_coords = value_amir['coords_list'][0]
        cur_bb_coords = get_backbone_coords(value_amir['coords_list'][0], key_amir)
        cur_c4p_coords = get_c4p_coords(value_amir['coords_list'][0])

    
        unpaired_idx = get_unpaird_amir(cur_c4p_coords.shape[0],cur_sec_struct)

        cur_unpaired = cur_c4p_coords[unpaired_idx]
        X = cur_unpaired
        if (len(X) > 10):
            nts['A'] += value_amir['sequence'].count('A')
            nts['C'] += value_amir['sequence'].count('C')
            nts['G'] += value_amir['sequence'].count('G')
            nts['U'] += value_amir['sequence'].count('U')
            for my_idx in unpaired_idx:
                 if value_amir['sequence'][my_idx] == 'A':
                      nts_unpaired['A']+=1
                 if value_amir['sequence'][my_idx] == 'C':
                      nts_unpaired['C']+=1  
                 if value_amir['sequence'][my_idx] == 'G':
                      nts_unpaired['G']+=1 
                 if value_amir['sequence'][my_idx] == 'U':
                      nts_unpaired['U']+=1 

            db = DBSCAN(eps=10, min_samples=5).fit(X)
            labels = db.labels_

            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            print("Estimated number of clusters: %d" % n_clusters_)
            print("Estimated number of noise points: %d" % n_noise_)

            unique_labels = set(labels)
            core_samples_mask = np.zeros_like(labels, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            
            

            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                 # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = labels == k
                masks = class_member_mask & core_samples_mask
                unpaired_pos = np.arange(0,X.shape[0])
                masks_idx = unpaired_pos[masks]
                x = X[masks]
                # x = X[class_member_mask]
                for i in range(0,len(x)-1):
                    for j in range(i+1,len(x)):
                        twist = get_twist_amir(x, i, j, masks_idx, unpaired_idx)
                        d = dist_amir(x[i], x[j])
                        c_i_1 = cur_bb_coords[unpaired_idx[masks_idx[i]]][1] - cur_bb_coords[unpaired_idx[masks_idx[i]]][0]
                        c_i_2 = cur_bb_coords[unpaired_idx[masks_idx[i]]][1] - cur_bb_coords[unpaired_idx[masks_idx[i]]][2]
                        c_j_1 = cur_bb_coords[unpaired_idx[masks_idx[j]]][1] - cur_bb_coords[unpaired_idx[masks_idx[j]]][0]
                        c_j_2 = cur_bb_coords[unpaired_idx[masks_idx[j]]][1] - cur_bb_coords[unpaired_idx[masks_idx[j]]][2]

                        c_i = normed_cross(c_i_1,c_i_2)
                        c_j = normed_cross(c_j_1,c_j_2)
                        phi = get_angle_amir(c_j,c_i)
                        theta = get_angle_amir(normed_vec(c_i_1),normed_vec(c_j_1))
                        n_i = value_amir['sequence'][unpaired_idx[masks_idx[i]]]
                        n_j = value_amir['sequence'][unpaired_idx[masks_idx[j]]]
                        if (twist > 100) and (2 < d < 20):
                            twists.append({'seq':value_amir['sequence'], 'X': cur_bb_coords, 'twist': twist, 'k':value_amir['k'], 
                            'masks_idx': masks_idx, 'unpaired_idx': unpaired_idx, 
                            'n_i': n_i, 'n_j': n_j, 'd':d, 'phi':phi,'theta':theta,
                            'i':unpaired_idx[masks_idx[i]], 'j': unpaired_idx[masks_idx[j]]})
                

    print(f"\nSaving (partially) processed data to {DATA_PATH}")
    torch.save(twists, os.path.join(DATA_PATH, "twists.pt"))


    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for entry in twists:
        cur_phi = entry['phi']
        cur_theta = entry['theta']
        cur_d = entry['d']
        if (entry['n_i'] == 'A') & (entry['n_j'] == 'A'):
             pass
        elif (entry['n_i'] == 'C') & (entry['n_j'] == 'C'):
             pass
        elif (entry['n_i'] == 'G') & (entry['n_j'] == 'G'):
             pass
        elif (entry['n_i'] == 'U') & (entry['n_j'] == 'U'):
             pass
        elif (entry['n_i'] == 'G') & (entry['n_j'] == 'A'):
             pass
        elif (entry['n_i'] == 'A') & (entry['n_j'] == 'G'):
                cur_phi *= -1; cur_theta *= -1
        elif (entry['n_i'] == 'G') & (entry['n_j'] == 'U'):
             pass
        elif (entry['n_i'] == 'U') & (entry['n_j'] == 'G'):
             cur_phi *= -1; cur_theta *= -1
        elif (entry['n_i'] == 'G') & (entry['n_j'] == 'C'):
             pass
        elif (entry['n_i'] == 'C') & (entry['n_j'] == 'G'):
             cur_phi *= -1; cur_theta *= -1
        elif (entry['n_i'] == 'U') & (entry['n_j'] == 'A'):
             pass
        elif (entry['n_i'] == 'A') & (entry['n_j'] == 'U'):
             cur_phi *= -1; cur_theta *= -1
        elif (entry['n_i'] == 'C') & (entry['n_j'] == 'A'):
             pass
        elif (entry['n_i'] == 'A') & (entry['n_j'] == 'C'):
             cur_phi *= -1; cur_theta *= -1

        cur_x, cur_y, cur_z = spheric2cartesian(cur_d, cur_theta, cur_phi)
        ax.scatter3D(cur_x, cur_y, cur_z, c='black')

    
    
    print('Nucleotide distribution')
    print(nts)

    print('Unpaired distribution')
    print(nts_unpaired)

   
    stop = True
