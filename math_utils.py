import numpy as np

def cos_sim(a, b):
    '''
    return cosyne similarity ([-1, 1]), and angle (radius)
    '''
    sim = np.clip(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)),  -1.0, 1.0)
    return (sim, np.arccos(sim))

def L2_distance(a, b):
    '''
    returns Eucledian distance
    '''
    return np.linalg.norm(a - b)

def shared_GCcount(GC_act):
    '''
    return shared # of GC for diff. odor inputs
    '''
    return np.all(GC_act, axis = 1).sum()