
import numpy as np
import os
import time
import pandas as pd

np.set_printoptions(suppress=True)

import plotly.graph_objects as go
from IPython.display import clear_output

from util import *

def Embryo_graph(n):
    
    adj = np.zeros((n,n),'int')
    
    for i in range(n):
        for j in range(n):

            if i-1 == j:
                adj[i,j]=1

            
            if i-2==j:
                adj[i,j]=1
                
            if i%2:
                if i-3==j:
                    adj[i,j]=1
                
    adj[-1,-2]=1
    
    edge_list = np.transpose(np.nonzero(adj))
                
    return adj, edge_list

# Misc functions
def update_progress(start, current_items, total_items, bar_length=50, label="Progress"):

    progress = (current_items-start)/(total_items - start)

    if progress < 0:
        progress = 0
    if progress >= 1 or total_items == current_items:
        progress = 1
        
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    # text = "{0}: [{1}] {2}/{3}".format(label, "#" * block + "-" * (bar_length - block), current_items+1, total_items)
    text = "{0}: [{1}]".format(label, "#" * block + "-" * (bar_length - block))
    print(text)

# Plotting functions
def create_edge_connections(adj_mat, coords):
    # a list of edge connections

    all_connections = []
    for row_idx in range(adj_mat.shape[0]):  # from node
        connections = []
        for col_idx in range(adj_mat.shape[1]):  # to node
            if adj_mat[row_idx, col_idx] == 1:
                connections.append((row_idx, col_idx))
        all_connections.extend(connections)

    Xe, Ye, Ze = ([], [], [])
    for e in all_connections:
        # x-coordinates of edge ends
        Xe += [coords[e[0]][0], coords[e[1]][0], None]
        Ye += [coords[e[0]][1], coords[e[1]][1], None]
        Ze += [coords[e[0]][2], coords[e[1]][2], None]

    return Xe, Ye, Ze
    
def plot_3d_overlay(prev_coords, next_coords, fig=None, errors={}, scale=(1,1,1)):
        
    padding = 5

    x_range = [min(prev_coords[:,0].min(), next_coords[:,0].min())-padding, 
               max(prev_coords[:,0].max(), next_coords[:,0].max())+padding]
    y_range = [min(prev_coords[:,1].min(), next_coords[:,1].min())-padding, 
               max(prev_coords[:,1].max(), next_coords[:,1].max())+padding]
    z_range = [min(prev_coords[:,2].min(), next_coords[:,2].min())-padding, 
               max(prev_coords[:,2].max(), next_coords[:,2].max())+padding]
    
    # Plotting  --------------------------------------------------------
    all_data = []
    COLORS = ["#F11", "#11F"]
    COORD_LABELS = ['x', 'y', 'z']
    mode = "markers+text"

    n = prev_coords.shape[0]

    if n == 19:
        names = ['H0L', 'H0R', 'H1L', 'H1R', 'H2L', 'H2R', 'V1L', 'V1R', 'V2L', 'V2R', 'V3L', 'V3R', 'V4L', 'V4R', 'V5L', 'V5R', 'V6L', 'V6R', 'T'] 
    elif n == 21:
        names = ['H0L', 'H0R', 'H1L', 'H1R', 'H2L', 'H2R', 'V1L', 'V1R', 'V2L', 'V2R', 'V3L', 'V3R', 'V4L', 'V4R', 'QL', 'QR', 'V5L', 'V5R', 'V6L', 'V6R', 'T'] 

    for data_idx, data in enumerate([prev_coords, next_coords]):
        
        for point_idx, point in enumerate(data):


            single_data_point = go.Scatter3d(
                x=[point[0]], 
                y=[point[1]], 
                z=[point[2]],
                marker={
                    "sizemode": "area",
                    "size": 10,
                    "color": COLORS[data_idx],
                    "line": {"width": 1, "color": "#000"}
                },
                mode=mode,
                text=names[point_idx],
                name=names[point_idx]
            )
            all_data.append(single_data_point)

            
        # lattice
        adj_mat, edge_list = Embryo_graph(n)
        edges = create_edge_connections(adj_mat, data)
        lattice = dict(
            x=edges[0],
            y=edges[1],
            z=edges[2],
            mode='lines',
            line=dict(color=COLORS[data_idx], width=10),
            hoverinfo='none',
            type='scatter3d',
        )
        all_data.append(lattice)
    
    if not fig:
        layout = {
            'template': "plotly_white",
            'scene': {
                'aspectmode': 'manual',
                'aspectratio':{'x':scale[0], 'y':scale[1], 'z':scale[2]},
                'xaxis': {
                    'showgrid': False,
                    "title": "X",
                    "range": x_range,
                },
                'yaxis': {
                    'showgrid': False,
                    "title": "Y",
                    "range": y_range # 0, 300
                    # "visible": False
                },
                'zaxis': {
                    'showgrid': False,
                    "title": "Z",
                    "range": z_range # 0, 300
                    # "visible": False
                },
            },
            'showlegend': False,
            'hovermode': "closest",
            'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
            'height': 500,
            'width': 900
        }

        # make figure
        fig_dict = {
            "data": all_data,
            "layout": layout,
        }
        fig = go.Figure(fig_dict)
    else: # if the figure already exists, use previous settings
        fig['data'] = all_data

    fig.show()

# MIPAV
def correction(vol_num, output_root, prev_coords, pred_coords):
    
    # prompt user
    resp = input("Do you want to correct the blue lattice? ([N]/Y): ")
    if "y" not in resp.lower():
        fixed_coords = pred_coords

        return fixed_coords
    
    # 
    in_path = os.path.join(output_root, 'corrections', 'corrections_'+str(vol_num)+'.csv')

    resp = ''
    while 'done' not in resp.lower():
        clear_output(wait = True)
        print('Place the corrected coordinates here:', in_path)
        resp = input("Type 'done' when finished editing: ")
        fixed_coords = pd.read_csv(in_path)
        fixed_coords = np.array(fixed_coords[['x', 'y','z']])
        plot_3d_overlay(prev_coords, fixed_coords)
    
    return fixed_coords

