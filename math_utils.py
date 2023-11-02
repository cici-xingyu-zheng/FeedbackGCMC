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

def active_GC_counts_overtime(GC_acts_overtime):
    
    return (GC_acts_overtime != 0).sum(axis = 1)

# def responsive_MCcount():
#     '''
#     for every responses[t]:
#     # either both fire, or diverge (is it neccessary even?)
#     response[t, :, 0] or response[t, :, 1] > 0
#     '''
    
#     pass

# def divergent_MCcount():
#     passm

