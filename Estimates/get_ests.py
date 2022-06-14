
import numpy as np
import os

np.set_printoptions(suppress=True)


def get_edge_list(n):
    
    W = np.zeros((n,n),'int')
    
    for i in range(n):
        for j in range(n):

            if i-1 == j:
                W[i,j]=1

            
            if i-2==j:
                W[i,j]=1
                
            if i%2:
                if i-3==j:
                    W[i,j]=1
                
    W[-1,-2]=1
    
    edge_list = np.transpose(np.nonzero(W))
                
    return W, edge_list

def get_dist_cost(d0_arr, d1_arr, edge_list):

    m = edge_list.shape[0]
    
    edge_costs = np.empty(m)
    
    for z in range(m):

        i, j = edge_list[z]

        xi = d0_arr[i] 
        xip = d1_arr[i] 
        xj = d0_arr[j] 
        xjp = d1_arr[j] 

        edge_ij = xj - xi # vector in first volume
        edge_ipjp = xjp - xip # vector in second vol

        edge_ij_len = np.linalg.norm(edge_ij)
        edge_ipjp_len = np.linalg.norm(edge_ipjp)

        edge_costs[z] = edge_ij_len - edge_ipjp_len

    return edge_costs

def get_nuclei_move(d0_arr, d1_arr):

    n = d0_arr.shape[0]
    nuclei_moves = np.empty((n, 3))
    
    for i in range(n):
                                    
        xi = d0_arr[i] # current node i in current volume
        xj = d1_arr[i] # connection from i->j in current volume

        edge_ij = xj - xi # vector in first volume

        nuclei_moves[i] = edge_ij 

    return nuclei_moves.flatten()

if __name__ == '__main__':
    
    home = os.path.join('/data', 'lauzierean', 'Evan_Tracking', 'Experiments')
    data_path = os.path.join(home, 'Data')
    est_path = os.path.join(home, 'Estimates')

    # First 10_12
    data_path_10_12 = os.path.join(data_path, '10_12')
    est_path_10_12 = os.path.join(est_path, '10_12')

    arrays = np.load(os.path.join(data_path_10_12, 'arrays_scaled.npy'))
    split, hatch = np.load(os.path.join(data_path_10_12, 'split_hatch.npy'))

    # Split into 19 / 21
    arrays_19 = arrays.copy()
    arrays_19 = np.delete(arrays_19, [14,15], axis=1)
    arrays_19 = arrays_19[:split]
    n_19 = arrays_19.shape[0]

    arrays_21 = arrays[split:hatch]
    n_21 = arrays_21.shape[0]

    # Save arrays
    np.save(os.path.join(data_path_10_12, 'arrays_scaled_10_12_0.npy'), arrays_19)
    np.save(os.path.join(data_path_10_12, 'arrays_scaled_10_12_1.npy'), arrays_21)

    W_19, edge_list_19 = get_edge_list(19)
    m_19 = edge_list_19.shape[0]

    W_21, edge_list_21 = get_edge_list(21)
    m_21 = edge_list_21.shape[0]

    # Movement
    nuclear_move_19 = np.empty((n_19-1,19))
    for i in range(1,n_19):
        nuclear_move_19[i-1] = np.linalg.norm(arrays_19[i]-arrays_19[i-1],axis=1)
        
    frame_move_19 = nuclear_move_19.sum(axis=1)

    low_t_19 = np.percentile(frame_move_19,50)
    mid_t_19 = np.percentile(frame_move_19,75)

    low_19 = frame_move_19<=low_t_19
    mid_19 = ((frame_move_19>=low_t_19) & (frame_move_19<=mid_t_19))
    high_19 = frame_move_19>=mid_t_19

    nuclear_move_21 = np.empty((n_21-1,21))
    for i in range(1,n_21):
        nuclear_move_21[i-1] = np.linalg.norm(arrays_21[i]-arrays_21[i-1],axis=1)
        
    frame_move_21 = nuclear_move_21.sum(axis=1)

    low_t_21 = np.percentile(frame_move_21,50)
    mid_t_21 = np.percentile(frame_move_21,75)

    low_21 = frame_move_21<=low_t_21
    mid_21 = ((frame_move_21>=low_t_21) & (frame_move_21<=mid_t_21))
    high_21 = frame_move_21>=mid_t_21

    # dist_full_cov
    edge_costs_19 = np.empty((n_19-1, m_19))
    for i in range(1, n_19):
        
        d0 = arrays_19[i-1]
        d1 = arrays_19[i]
        
        ecs = get_dist_cost(d0, d1, edge_list_19)
        
        edge_costs_19[i-1] = ecs

    edge_cov_19 = np.cov(edge_costs_19,rowvar=0)
    edge_cov_19_inv = np.linalg.inv(edge_cov_19)

    edge_costs_19_low = edge_costs_19[low_19]
    edge_costs_19_mid = edge_costs_19[mid_19]
    edge_costs_19_high = edge_costs_19[high_19]

    edge_cov_19_low = np.cov(edge_costs_19_low,rowvar=0)
    edge_cov_19_low_inv = np.linalg.inv(edge_cov_19_low)

    edge_cov_19_mid = np.cov(edge_costs_19_mid,rowvar=0)
    edge_cov_19_mid_inv = np.linalg.inv(edge_cov_19_mid)

    edge_cov_19_high = np.cov(edge_costs_19_high,rowvar=0)
    edge_cov_19_high_inv = np.linalg.inv(edge_cov_19_high)

    # 21
    edge_costs_21 = np.empty((n_21-1, m_21))
    for i in range(1, n_21):
        
        d0 = arrays_21[i-1]
        d1 = arrays_21[i]
        
        ecs = get_dist_cost(d0, d1, edge_list_21)
        
        edge_costs_21[i-1] = ecs

    edge_cov_21 = np.cov(edge_costs_21,rowvar=0)
    edge_cov_21_inv = np.linalg.inv(edge_cov_21)

    edge_costs_21_low = edge_costs_21[low_21]
    edge_costs_21_mid = edge_costs_21[mid_21]
    edge_costs_21_high = edge_costs_21[high_21]

    edge_cov_21_low = np.cov(edge_costs_21_low,rowvar=0)
    edge_cov_21_low_inv = np.linalg.inv(edge_cov_21_low)

    edge_cov_21_mid = np.cov(edge_costs_21_mid,rowvar=0)
    edge_cov_21_mid_inv = np.linalg.inv(edge_cov_21_mid)

    edge_cov_21_high = np.cov(edge_costs_21_high,rowvar=0)
    edge_cov_21_high_inv = np.linalg.inv(edge_cov_21_high)

    # Move corr
    linear_moves_19 = np.empty((n_19-1, 19*3))

    for i in range(1, n_19):
        
        d0 = arrays_19[i-1]
        d1 = arrays_19[i]
        
        nuclei_moves = get_nuclei_move(d0,d1)
        linear_moves_19[i-1] = nuclei_moves

    linear_moves_19_cov = np.cov(linear_moves_19,rowvar=0)
    linear_moves_19_cov_inv = np.linalg.inv(linear_moves_19_cov)

    # Low, Med, High
    linear_moves_19_low = linear_moves_19[low_19]
    linear_moves_19_mid = linear_moves_19[mid_19]
    linear_moves_19_high = linear_moves_19[high_19]

    linear_moves_19_low_cov = np.cov(linear_moves_19_low,rowvar=0)
    linear_moves_19_low_cov_inv = np.linalg.inv(linear_moves_19_low_cov)

    linear_moves_19_mid_cov = np.cov(linear_moves_19_mid,rowvar=0)
    linear_moves_19_mid_cov_inv = np.linalg.inv(linear_moves_19_mid_cov)

    linear_moves_19_high_cov = np.cov(linear_moves_19_high,rowvar=0)
    linear_moves_19_high_cov_inv = np.linalg.inv(linear_moves_19_high_cov)

    # 21

    linear_moves_21 = np.empty((n_21-1, 21*3))

    for i in range(1, n_21):
        
        d0 = arrays_21[i-1]
        d1 = arrays_21[i]
        
        # print(d0.shape, d1.shape)
        nuclei_moves = get_nuclei_move(d0,d1)
        linear_moves_21[i-1] = nuclei_moves

    linear_moves_21_cov = np.cov(linear_moves_21,rowvar=0)
    linear_moves_21_cov_inv = np.linalg.inv(linear_moves_21_cov)

    # Low, Med, High
    linear_moves_21_low = linear_moves_21[low_21]
    linear_moves_21_mid = linear_moves_21[mid_21]
    linear_moves_21_high = linear_moves_21[high_21]

    linear_moves_21_low_cov = np.cov(linear_moves_21_low,rowvar=0)
    linear_moves_21_low_cov_inv = np.linalg.inv(linear_moves_21_low_cov)

    linear_moves_21_mid_cov = np.cov(linear_moves_21_mid,rowvar=0)
    linear_moves_21_mid_cov_inv = np.linalg.inv(linear_moves_21_mid_cov)

    linear_moves_21_high_cov = np.cov(linear_moves_21_high,rowvar=0)
    linear_moves_21_high_cov_inv = np.linalg.inv(linear_moves_21_high_cov)

    # Now save ALL
    np.save(os.path.join(est_path_10_12, 'quad_inv_cov_10_12_0.npy'), edge_cov_19_inv)
    np.save(os.path.join(est_path_10_12, 'quad_inv_cov_low_10_12_0.npy'), edge_cov_19_low_inv)
    np.save(os.path.join(est_path_10_12, 'quad_inv_cov_mid_10_12_0.npy'), edge_cov_19_mid_inv)
    np.save(os.path.join(est_path_10_12, 'quad_inv_cov_high_10_12_0.npy'), edge_cov_19_high_inv)

    np.save(os.path.join(est_path_10_12, 'linear_moves_inv_cov_10_12_0.npy'), linear_moves_19_cov_inv)
    np.save(os.path.join(est_path_10_12, 'linear_moves_inv_cov_low_10_12_0.npy'), linear_moves_19_low_cov_inv)
    np.save(os.path.join(est_path_10_12, 'linear_moves_inv_cov_mid_10_12_0.npy'), linear_moves_19_mid_cov_inv)
    np.save(os.path.join(est_path_10_12, 'linear_moves_inv_cov_high_10_12_0.npy'), linear_moves_19_high_cov_inv)

    np.save(os.path.join(est_path_10_12, 'quad_inv_cov_10_12_1.npy'), edge_cov_21_inv)
    np.save(os.path.join(est_path_10_12, 'quad_inv_cov_low_10_12_1.npy'), edge_cov_21_low_inv)
    np.save(os.path.join(est_path_10_12, 'quad_inv_cov_mid_10_12_1.npy'), edge_cov_21_mid_inv)
    np.save(os.path.join(est_path_10_12, 'quad_inv_cov_high_10_12_1.npy'), edge_cov_21_high_inv)

    np.save(os.path.join(est_path_10_12, 'linear_moves_inv_cov_10_12_1.npy'), linear_moves_21_cov_inv)
    np.save(os.path.join(est_path_10_12, 'linear_moves_inv_cov_low_10_12_1.npy'), linear_moves_21_low_cov_inv)
    np.save(os.path.join(est_path_10_12, 'linear_moves_inv_cov_mid_10_12_1.npy'), linear_moves_21_mid_cov_inv)
    np.save(os.path.join(est_path_10_12, 'linear_moves_inv_cov_high_10_12_1.npy'), linear_moves_21_high_cov_inv)

    # Now 04_06
    data_path_04_06 = os.path.join(data_path, '04_06')
    est_path_04_06 = os.path.join(est_path, '04_06')

    np.save(os.path.join(est_path_04_06, 'quad_inv_cov_04_06_1.npy'), edge_cov_21_inv)
    np.save(os.path.join(est_path_04_06, 'quad_inv_cov_low_04_06_1.npy'), edge_cov_21_low_inv)
    np.save(os.path.join(est_path_04_06, 'quad_inv_cov_mid_04_06_1.npy'), edge_cov_21_mid_inv)
    np.save(os.path.join(est_path_04_06, 'quad_inv_cov_high_04_06_1.npy'), edge_cov_21_high_inv)

    np.save(os.path.join(est_path_04_06, 'linear_moves_inv_cov_04_06_1.npy'), linear_moves_21_cov_inv)
    np.save(os.path.join(est_path_04_06, 'linear_moves_inv_cov_low_04_06_1.npy'), linear_moves_21_low_cov_inv)
    np.save(os.path.join(est_path_04_06, 'linear_moves_inv_cov_mid_04_06_1.npy'), linear_moves_21_mid_cov_inv)
    np.save(os.path.join(est_path_04_06, 'linear_moves_inv_cov_high_04_06_1.npy'), linear_moves_21_high_cov_inv)

    arrays = np.load(os.path.join(data_path_04_06, 'arrays_scaled.npy'))
    split, hatch = np.load(os.path.join(data_path_04_06, 'split_hatch.npy'))

    # Split into 19 / 21
    arrays_19 = arrays.copy()
    # arrays_19 = np.delete(arrays_19, [14,15], axis=1)
    arrays_19 = arrays_19[:split]
    n_19 = arrays_19.shape[0]

    # Save arrays
    np.save(os.path.join(data_path_04_06, 'arrays_scaled_04_06_0.npy'), arrays_19)

    W_19, edge_list_19 = get_edge_list(19)
    m_19 = edge_list_19.shape[0]

    # Movement
    nuclear_move_19 = np.empty((n_19-1,19))
    for i in range(1,n_19):
        nuclear_move_19[i-1] = np.linalg.norm(arrays_19[i]-arrays_19[i-1],axis=1)
        
    frame_move_19 = nuclear_move_19.sum(axis=1)

    low_t_19 = np.percentile(frame_move_19,50)
    mid_t_19 = np.percentile(frame_move_19,75)

    low_19 = frame_move_19<=low_t_19
    mid_19 = ((frame_move_19>=low_t_19) & (frame_move_19<=mid_t_19))
    high_19 = frame_move_19>=mid_t_19

    # dist_full_cov
    edge_costs_19 = np.empty((n_19-1, m_19))
    for i in range(1, n_19):
        
        d0 = arrays_19[i-1]
        d1 = arrays_19[i]
        
        ecs = get_dist_cost(d0, d1, edge_list_19)
        
        edge_costs_19[i-1] = ecs

    edge_cov_19 = np.cov(edge_costs_19,rowvar=0)
    edge_cov_19_inv = np.linalg.inv(edge_cov_19)

    edge_costs_19_low = edge_costs_19[low_19]
    edge_costs_19_mid = edge_costs_19[mid_19]
    edge_costs_19_high = edge_costs_19[high_19]

    edge_cov_19_low = np.cov(edge_costs_19_low,rowvar=0)
    edge_cov_19_low_inv = np.linalg.inv(edge_cov_19_low)

    edge_cov_19_mid = np.cov(edge_costs_19_mid,rowvar=0)
    edge_cov_19_mid_inv = np.linalg.inv(edge_cov_19_mid)

    edge_cov_19_high = np.cov(edge_costs_19_high,rowvar=0)
    edge_cov_19_high_inv = np.linalg.inv(edge_cov_19_high)

    # Move corr
    linear_moves_19 = np.empty((n_19-1, 19*3))

    for i in range(1, n_19):
        
        d0 = arrays_19[i-1]
        d1 = arrays_19[i]
        
        nuclei_moves = get_nuclei_move(d0,d1)
        linear_moves_19[i-1] = nuclei_moves

    linear_moves_19_cov = np.cov(linear_moves_19,rowvar=0)
    linear_moves_19_cov_inv = np.linalg.inv(linear_moves_19_cov)

    # Low, Med, High
    linear_moves_19_low = linear_moves_19[low_19]
    linear_moves_19_mid = linear_moves_19[mid_19]
    linear_moves_19_high = linear_moves_19[high_19]

    linear_moves_19_low_cov = np.cov(linear_moves_19_low,rowvar=0)
    linear_moves_19_low_cov_inv = np.linalg.inv(linear_moves_19_low_cov)

    linear_moves_19_mid_cov = np.cov(linear_moves_19_mid,rowvar=0)
    linear_moves_19_mid_cov_inv = np.linalg.inv(linear_moves_19_mid_cov)

    linear_moves_19_high_cov = np.cov(linear_moves_19_high,rowvar=0)
    linear_moves_19_high_cov_inv = np.linalg.inv(linear_moves_19_high_cov)

    # Now save ALL
    np.save(os.path.join(est_path_04_06, 'quad_inv_cov_04_06_0.npy'), edge_cov_19_inv)
    np.save(os.path.join(est_path_04_06, 'quad_inv_cov_low_04_06_0.npy'), edge_cov_19_low_inv)
    np.save(os.path.join(est_path_04_06, 'quad_inv_cov_mid_04_06_0.npy'), edge_cov_19_mid_inv)
    np.save(os.path.join(est_path_04_06, 'quad_inv_cov_high_04_06_0.npy'), edge_cov_19_high_inv)

    np.save(os.path.join(est_path_04_06, 'linear_moves_inv_cov_04_06_0.npy'), linear_moves_19_cov_inv)
    np.save(os.path.join(est_path_04_06, 'linear_moves_inv_cov_low_04_06_0.npy'), linear_moves_19_low_cov_inv)
    np.save(os.path.join(est_path_04_06, 'linear_moves_inv_cov_mid_04_06_0.npy'), linear_moves_19_mid_cov_inv)
    np.save(os.path.join(est_path_04_06, 'linear_moves_inv_cov_high_04_06_0.npy'), linear_moves_19_high_cov_inv)
