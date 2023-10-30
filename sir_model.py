# Implement the sparse incomplete GC representation model: for a given input,
# with an fixed MC-GC network, find the odor representation by minimizing the 
# loss function using backtracking line search. 

# end of 2023 will try to make a class of the network, then it'd be nice to define 
# descending methods on different attributes.

import numpy as np
import scipy as sc
# import copy

import utils

# shared helper functions:

def get_err(GC_act, odor_input, W):

    return odor_input - np.matmul(W, GC_act)


def get_loss(GC_act, odor_input, theta, W):
    '''
    Inputs:
    1) gc activities
    2) net mc activity, r_m
    3) gc threshold, theta

    Output: loss value
    '''
    MC_err = get_err(GC_act, odor_input, W)
    loss = (1/2)*(sc.linalg.norm(MC_err, 2)**2) + theta*np.sum(GC_act)
    
    return loss

def get_gradient(MC_err, theta, W):
    '''
    Inputs:
    1) gc activities
    2) net mc activity, r_m
    3) MC-GC network

    Function: gradient of the loss function w.r.t gc activations
    '''
  
    grad = - np.matmul(W.T, MC_err) + theta

    return grad

def generalized_grad(GC_iter, grad, theta, t):
    '''
    Proximal gradient computation 

    Inputs: 
    1) iterate: current iterate
    2) grad: gradient at current iterate
    3) theta: gc threshold
    4) t: steplength in [0,1]

    Output: proximal gradient
    '''
    GC_aftergrad =  utils.project(GC_iter - t*grad, theta)

    return GC_iter - GC_aftergrad

# Quasi-Newton's method:

def newtons_update(GC_act, W, projected_grad):

    pseudo_hess = np.linalg.pinv(np.matmul(W.T, W))
    # get pseudo_hessian:
    d = - pseudo_hess @ projected_grad
    GC_act += d
    return GC_act


alpha = .5 # \in (0, 0.5)
beta = .8 # \in (0, 1)
iters = 50
def line_search_update(grad, GC_act, odor_input, theta, curr_loss, W): 
    t = 1
    gen_grad = generalized_grad(GC_act, grad, theta, t)
    new_loss =  get_loss(GC_act - gen_grad, odor_input, theta, W)
    # Armijo_bool = new_loss > curr_loss - alpha*t*(np.dot(grad, grad)) # I see grad*grad != generalized_grad* generalized_grad
    # a modification of the sufficient descent creteria: 
    Armijo_bool = new_loss > curr_loss - alpha*t*(np.dot(gen_grad, gen_grad)) 

    count = 0
    
    while Armijo_bool and count < iters:
        t *= beta
        curr_loss = new_loss
        gen_grad = generalized_grad(GC_act, grad, theta, t)
        new_loss =  get_loss(GC_act - gen_grad, odor_input, theta, W)
        # Armijo_bool =  new_loss > curr_loss - alpha*t*(np.dot(grad, grad)) 
        Armijo_bool = new_loss > curr_loss - curr_loss - alpha*t*(np.dot(gen_grad, gen_grad)) 
        count += 1

    GC_act -= gen_grad
    
    return GC_act


## Frank-Wolfe Algorithm:
## https://fa.bianp.net/blog/2022/adaptive_fw/

beta = .9
iters = 50

def line_search_PGD_update(GC_act, odor_input, theta, grad, curr_loss, W):
    '''
    Backtracking line search for projected gradient descent

    Inputs: 
    1) gc activities
    2) net mc activities, r_m
    3) theta: gc threshold
    4) W: MC-GC network
    5) grad: gradient at current iterate (current gc activities)
    6) loss_curr: loss function value at current iterate
    7) gamma: line search parameter 

    Output: steplength 
    '''
    t = 1
    for _ in range(iters):
        gen_grad = generalized_grad(GC_act, grad, theta, t)
        # print('Norm of generalized gradient', sc.linalg.norm(gen_grad, 2))
        new_iterate = GC_act - t*gen_grad
        new_loss = get_loss(new_iterate, odor_input, theta, W)

        ### I don't understand this...
        quad_approx = curr_loss - (t*np.matmul(grad, np.transpose(gen_grad))) + (t/2)*(sc.linalg.norm(gen_grad, 2)**2)
        
        if new_loss < curr_loss and new_loss <= quad_approx:
            break 
        else:
            t = beta*t # backtrack till objective value at new point is smaller than a quadratic approximation
    
    # maybe haveing the zero cap make it easier:    
    GC_act -= t*generalized_grad(GC_act, gen_grad, theta, t)   

    return GC_act


def sniff_cycle(odor_input, GC_act, theta, W):    
    
    for _ in range(1000):
        MC_err = odor_input - np.matmul(W, GC_act)
        grad = get_gradient(MC_err, theta, W)
        # does normalizing it make it easier to serach t?
        norm_grad = grad/sc.linalg.norm(grad,2)  
        # can be swapped with other _update functions:
        loss = get_loss(GC_act, odor_input, theta, W)
        GC_act = line_search_PGD_update(GC_act, odor_input, theta, norm_grad, loss, W)
        
    return GC_act


# regularized updates: Plasticity
def hebbian_update(W, GC_act, odor_input, etas, cap = True, cap_strength = 1):
    for a in range(len(GC_act)):
        for i in range(len(odor_input)):
            if GC_act[a] > 0 and odor_input[i] > 0:
                W[i,a] = min(W[i,a] + etas['associate']*odor_input[i]*GC_act[a], 1) 
            # can perhaps try not updating:
            elif GC_act[a] > 0 and odor_input[i] < 0:
                W[i,a] = max(W[i,a] + etas['disassociate']*odor_input[i]*GC_act[a], 0)
            else:
                W[i,a] = max(W[i,a] - etas['forget'], 0)
        if cap:
            if np.sum(W[:,a]) > cap_strength:
                W[:,a] = (cap_strength*W[:,a])/np.sum(W[:,a])