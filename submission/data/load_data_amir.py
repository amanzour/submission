import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import dotenv
dotenv.load_dotenv(".env")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import math

import os
import argparse
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import torch
import torch_cluster

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import biotite
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.analysis.rms import rmsd as get_rmsd

from src.data.data_utils import pdb_to_tensor, pdb_to_tensor_amir, get_twist_amir, get_c4p_coords, get_backbone_coords, dist_amir
from src.data.clustering_utils import cluster_sequence_identity, cluster_structure_similarity
from src.data.sec_struct_utils import pdb_to_x3dna_amir, x3dna_to_sec_struct_amir,get_unpaird_amir
from src.data.featurizer import normed_vec, normed_cross, spheric2cartesian_amir
from scipy.spatial.distance import cdist
from matplotlib import cm


import warnings
warnings.filterwarnings("ignore", category=biotite.structure.error.IncompleteStructureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

DATA_PATH = os.environ.get("DATA_PATH")

keep_insertions = True


if __name__ == "__main__":
    
    #twists = torch.load(os.path.join(DATA_PATH, "twists.pt"))
    processed = list(torch.load(os.path.join(DATA_PATH, "processed.pt")).values())

    nts = {'A': 676854, 'C': 649106, 'G': 844054, 'U': 555949}
    nts_unpaired = {'A': 242489, 'C': 120123, 'G': 150241, 'U': 151962}

    pairs_df = pd.DataFrame(columns = ['phi', 'theta', 'd', 'pair','i','j','k','x','y','z']) 

    pairs = {'AA':[],'CC':[],'GG':[],'UU':[],'GA':[],'GU':[],'GC':[],'UA':[],'UC':[], 'CA':[]}
    
    for entry in processed:
         print(entry['sec_bp_list'])
         
    for entry in twists:
        cur_phi = abs(entry['phi'])
        cur_theta = entry['theta']
        cur_d = entry['d']
        cur_pair = entry['n_i']+entry['n_j']
        cur_i = entry['i']
        cur_j = entry['j']
        cur_k = entry['k']
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
             cur_pair=entry['n_j']+entry['n_i']; cur_theta = -entry['theta']
        elif (entry['n_i'] == 'G') & (entry['n_j'] == 'U'):
             pass
        elif (entry['n_i'] == 'U') & (entry['n_j'] == 'G'):
             cur_pair=entry['n_j']+entry['n_i']; cur_theta = -entry['theta']
        elif (entry['n_i'] == 'G') & (entry['n_j'] == 'C'):
             pass
        elif (entry['n_i'] == 'C') & (entry['n_j'] == 'G'):
             cur_pair=entry['n_j']+entry['n_i']; cur_theta = -entry['theta']
        elif (entry['n_i'] == 'U') & (entry['n_j'] == 'A'):
             pass
        elif (entry['n_i'] == 'A') & (entry['n_j'] == 'U'):
             cur_pair=entry['n_j']+entry['n_i']; cur_theta = -entry['theta']
        elif (entry['n_i'] == 'C') & (entry['n_j'] == 'A'):
             pass
        elif (entry['n_i'] == 'A') & (entry['n_j'] == 'C'):
             cur_pair=entry['n_j']+entry['n_i']; cur_theta = -entry['theta']
        
        cur_x, cur_y, cur_z = spheric2cartesian_amir(cur_d, cur_theta, cur_phi)/cur_d.numpy()
        cur_entry = data = {'phi': cur_phi.numpy(),
                            'theta': cur_theta.numpy(),
                            'd': cur_d.numpy(),
                            'pair': cur_pair,
                            'i': cur_i,
                            'j': cur_j,
                            'k': cur_k,
                            'x': cur_x,
                            'y': cur_y,
                            'z': cur_z }
        pairs_df = pd.concat([pairs_df, pd.DataFrame(cur_entry)]) 
        
    # density plot   
    def near( p, pntList, d0 ):
         cnt=0
         for pj in pntList:
              dist=np.linalg.norm( p - pj )
              if dist < d0:
                   cnt += 1 - dist/d0
         return cnt
    
    # density plot

#     pointList = np.array(10*pairs_df[['x', 'y', 'z']])
#     u = np.linspace( 0, 2 * np.pi, 120)
#     v = np.linspace( 0, np.pi, 60 )

#     XX = 10 * np.outer( np.cos( u ), np.sin( v ) )
#     YY = 10 * np.outer( np.sin( u ), np.sin( v ) )
#     ZZ = 10 * np.outer( np.ones( np.size( u ) ), np.cos( v ) )

#     WW = XX.copy()
#     for i in range( len( XX ) ):
#          for j in range( len( XX[0] ) ):
#               x = XX[ i, j ]
#               y = YY[ i, j ]
#               z = ZZ[ i, j ]
#               WW[ i, j ] = near(np.array( [x, y, z ] ), pointList, 3)

#     WW = WW / np.amax( WW )
#     myheatmap = WW
    
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.plot_surface( XX, YY,  ZZ, cstride=1, rstride=1, facecolors=cm.jet( myheatmap ) )
#     plt.show() 

    # elbow plot
    X = pairs_df[['x', 'y', 'z']]
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 10)
    for k in K:
         # Building and fitting the model
         kmeanModel = KMeans(n_clusters=k).fit(X)
         kmeanModel.fit(X)
         distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
         inertias.append(kmeanModel.inertia_)
         mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]
         mapping2[k] = kmeanModel.inertia_


    for key, val in mapping1.items():
         print(f'{key} : {val}')

#     plt.plot(K, distortions, 'bx-')
#     plt.xlabel('No. of clusters')
#     plt.ylabel('Distortion')
#     plt.title('The Elbow Method using Distortion')
#     plt.show()
    
    
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('No. of clusters')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()     



    model = KMeans(n_clusters = 3, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    clusters = model.fit_predict(pairs_df[['x', 'y', 'z']])

    cluster_max_index = -1
    cluster_max_count = 0
    print("cluster indices and sizes:")

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')  
    
    for index in set(clusters):
         print(index)
         print('length=',len(clusters[clusters == index]))
         cur_pairs = pairs_df[clusters == cluster_max_index]
         #ax.scatter(cur_pairs['x'], cur_pairs['y'], cur_pairs['z'], color = "gray", alpha=0.1)
         if len(clusters[clusters == index]) > cluster_max_count:
              cluster_max_count = len(clusters[clusters == index])
              cluster_max_index = index

    best_pairs = pairs_df[clusters == cluster_max_index]
    mean_vect = np.mean(best_pairs[['x','y','z']], axis=0)
    print(mean_vect)
    # ax.scatter(cur_pairs['x'], cur_pairs['y'], cur_pairs['z'], color = "green")

    # sns.countplot(clusters)
#     for this_pair in pairs.keys():
#          print(this_pair)
#          print((pairs_df[clusters == cluster_max_index]['pair']== this_pair).sum())
       
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xLabel = ax.set_xlabel('x')
    yLabel = ax.set_ylabel('y')
    zLabel = ax.set_zlabel('z')
    colors = np.array(['black' for r in range(pairs_df.shape[0])])
    colors[clusters == 0] = 'red'; colors[clusters == 1] = 'blue'; colors[clusters == 2] = 'green'
    print("best cluster is: ",cluster_max_index," with color ", colors[cluster_max_index])


    ax.scatter3D(pairs_df['x'], pairs_df['y'], pairs_df['z'], c=colors, alpha = 0.1)

    fig.show()
    mean_vect_s = pd.DataFrame(mean_vect)
    mult_vect = pairs_df[['x','y','z']].dot(mean_vect_s)
    best_real_vect_idx = np.argmax(mult_vect.values)
    best_real_vect = pairs_df.iloc[best_real_vect_idx]

    counter = 0
    entry_6ZVK = ""
    indicies_6ZVK = []
    dist_6ZVK = []
    for entry in twists:
        if ((entry['n_i'] == 'A') & (entry['n_j'] == 'A')) | ((entry['n_i'] == 'C') & (entry['n_j'] == 'U')):
             dot_product = entry['dir'][0]*best_real_vect['x'] + entry['dir'][1]*best_real_vect['y'] + entry['dir'][2]*best_real_vect['z']
             if dot_product > 0.99:
                  print('new entry: '+ ' at product '+str(dot_product))
                  print(entry['k'])
                  print(entry['twist'])
                  print(entry['n_i']+' at '+str(entry['i']))
                  print(entry['n_j']+' at '+str(entry['j']))
                  print('distance= '+str(entry['d'])+' direction= '+str(entry['dir']))
                  counter+=1
                  
                  if entry['k'][0] == '6ZVK_1_e2-h2':
                       entry_6ZVK = entry
                       indicies_6ZVK.append([entry['i'], entry['j']])
                       dist_6ZVK.append(entry['d'])
     
    print(counter)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    triangles = []
    triangles_nt = []
    triangles_pos = []
    for c,ind_i_j in enumerate(indicies_6ZVK):
         x_i = entry_6ZVK['X'][ind_i_j[0]]
         x_j = entry_6ZVK['X'][ind_i_j[1]]
         print(str(ind_i_j[0])," and ",str(ind_i_j[1]))
         ax.scatter(x_i[0,0], x_i[0,1], x_i[0,2], color = "green")
         ax.scatter(x_i[1,0], x_i[1,1], x_i[1,2], color = "red")
         ax.scatter(x_i[2,0], x_i[2,1], x_i[2,2], color = "blue")
         ax.text(x_i[0,0], x_i[0,1], x_i[0,2],  entry_6ZVK['seq'][ind_i_j[0]], size=10, zorder=1, color='k')
         ax.text(x_i[2,0], x_i[2,1], x_i[2,2],  str(ind_i_j[0]), size=10, zorder=1, color='k')

         ax.scatter(x_j[0,0], x_j[0,1], x_j[0,2], color = "green", alpha = 0.3)
         ax.scatter(x_j[1,0], x_j[1,1], x_j[1,2], color = "red", alpha = 0.3)
         ax.scatter(x_j[2,0], x_j[2,1], x_j[2,2], color = "blue", alpha = 0.3)
         ax.text(x_j[0,0], x_j[0,1], x_j[0,2],  entry_6ZVK['seq'][ind_i_j[1]], size=10, zorder=1, color='k')
         ax.text(x_j[2,0], x_j[2,1], x_j[2,2],  str(ind_i_j[1]), size=10, zorder=1, color='k')
    
         ax.plot3D([x_i[1,0],x_j[1,0]], [x_i[1,1],x_j[1,1]],[x_i[1,2],x_j[1,2]], 'maroon', alpha = 0.2, linestyle="dashed")
         ax.text((x_j[1,0]+x_i[1,0])/2, (x_j[1,1]+x_i[1,1])/2, (x_j[1,2]+x_i[1,2])/2, str(round(dist_6ZVK[c].item(),1)), size=10, color='maroon')

         triangles.append((x_i[0].numpy(),x_i[1].numpy(),x_i[2].numpy()))
         triangles.append((x_j[0].numpy(),x_j[1].numpy(),x_j[2].numpy()))
         triangles_nt.append(entry_6ZVK['seq'][ind_i_j[0]])
         triangles_nt.append(entry_6ZVK['seq'][ind_i_j[1]])

         triangles_pos.append(ind_i_j[0])
         triangles_pos.append(ind_i_j[1])
    # ax = plt.gca(projection="3d")
    fig.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    ax.add_collection(Poly3DCollection(triangles))
    ax.set_xlim([-40,-20])
    ax.set_ylim([-20,20])
    ax.set_zlim([-20,0])

    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    fig.show()
    stop = True


    
    for this_pair,this_vecs in pairs.items():
         print('pair:',this_pair)
         print('length:',str(len(this_vecs)))
         print((np.mean(this_vecs, axis=0)))
         print((np.std(this_vecs, axis=0)))
         print("")
