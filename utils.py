import numpy as np
from scipy.stats import bernoulli
from numpy.typing import NDArray


def create_network(M:int, 
                   N:int, 
                   p_conn:float, 
                   max_lim:float, 
                   seed = 2023,
                   cap = True, 
                   cap_strength = 1, 
                   verbose = True) -> NDArray:
    '''
    Create a random MC-GC network with given parameters. Each weight is drawn from
    a Bernoulli(0, max_lim).

    Inputs:
        M: number of MCs.
        N: number of GCs.
        p_conn: probability of connection.
        max_lim: Bernoulli right end.

    Output: 
        W: the connection (weight) matrix (M, N).
    '''

    np.random.seed(seed)

    W = np.zeros((M,N))  
    for i in range(M):
        for a in range(N):
            # to set the random seed for both numpy and scipy.stats:
            W[i,a] = bernoulli.rvs(p_conn, random_state = seed+i*N+a)*np.random.uniform(0,max_lim)

    # capping GC max connection strength to 1:
    if cap:   
        for a in range(N):
            # if GC's total connect exceeds reg:
            if np.sum(W[:, a]) > cap_strength: 
                W[:, a] = (cap_strength*W[:, a])/np.sum(W[:, a])
    
    # GC stats report (# of connected MC and strength)
    if verbose:
        columns = (W != 0).sum(axis = 0)
        column_sum = W.sum(axis = 0)     
        print('Max number of non zero GC connections', np.max(columns))
        print('Average number of non zero GC connections', np.mean(columns)) 
        print('Max strength of GC connections', np.max(column_sum))
        print('Average strength of GC connections', np.mean(column_sum))

    return W


def project(a, theta):
    '''
    Inputs:
    1) gc activities
    2) gc threshold

    Output: projection on the non-negative orthant via thresholding
    '''
    a[a <= theta] = 0
    
    return a

def get_GCact(W:NDArray, MC_act:NDArray, theta:float) -> NDArray:
    '''
    GC's threshholded activity given MC input.
    input: 
        W: (M, N)
        MC_act: (M, K)
    output:
        GC_act: (N, K)
    '''
    # thresholding:
    GC_act = project(np.matmul(W.T, MC_act), theta)
    
    return GC_act


