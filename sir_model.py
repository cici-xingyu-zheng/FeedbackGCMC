# Implement the sparse incomplete GC representation model: for a given input,
# with an fixed MC-GC network, find the odor representation by minimizing the 
# loss function using backtracking line search. 

import numpy as np
import scipy as sc
import copy

def project(z, theta):
    '''
    Inputs:
    1) gc activities
    2) gc threshold

    Output: projection on the non-negative orthant via thresholding
    '''
    z[z <= theta] = 0
    
    return z

def get_gradient(activities, net_mc, theta, W):
    '''
    Inputs:
    1) gc activities
    2) net mc activity, r_m
    3) MC-GC network

    Function: gradient of the loss function w.r.t gc activations
    '''
    grad = np.zeros((activities.size)) 
    for i in range(activities.size):
        grad[i] = - np.matmul(net_mc, W[i,:]) + theta

    return grad




def get_loss(activities, net_mc, theta):
    '''
    Inputs:
    1) gc activities
    2) net mc activity, r_m
    3) gc threshold, theta

    Output: loss value
    '''
    loss = (sc.linalg.norm(net_mc, 2)**2) + theta*np.sum(activities)
    
    return loss


# why we need this function?
def generalized_grad(iterate, grad, theta, t):
    '''
    Proximal gradient computation 

    Inputs: 
    1) iterate: current iterate
    2) grad: gradient at current iterate
    3) theta: gc threshold
    4) t: steplength in [0,1]

    Output: proximal gradient
    '''
    return (iterate - project(iterate - t*grad, theta))/t


# optimal step size 
def line_search_PGD(activities, net_mc, theta, W, grad, loss_curr, gamma):
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
    iter = 50
    for i in range(iter):
        gen_grad = generalized_grad(activities, grad, theta, t)
        # print('Norm of generalized gradient', sc.linalg.norm(gen_grad, 2))
        new_iterate = activities - t*gen_grad
        new_loss = get_loss(new_iterate, net_mc, theta)
        quad_approx = loss_curr - (t*np.matmul(grad, np.transpose(gen_grad))) + (t/2)*(sc.linalg.norm(gen_grad, 2)**2)
        
        if new_loss < loss_curr and new_loss <= quad_approx:
        # third_part = (t/2)*(sc.linalg.norm(gen_grad, 2)**2)
        # print('Quadratic approximation', quad_approx)
        # print('Current loss', loss_curr)
        # print('New loss', new_loss)
        # print('Second part', (t*np.matmul(grad, np.transpose(gen_grad))))
        # print('Third part', third_part)
        # print('t times Third part', t*third_part)
        # if quad_approx < loss_curr and new_loss <= quad_approx:
        # if new_loss < quad_approx:
            # print('Loss current', loss_curr)
            # print('New loss', new_loss)
            # print('Local upper bound', quad_approx)
            break 
        else:
            t = gamma*t # backtrack till objective value at new point is smaller than a quadratic approximation
    
    return t if i < iter - 1 else 0



def get_mc_backtrack_PGD(theta, mc_input, gc_init, num_steps, gamma, W, thresh):
    '''
    Inner loop function: For each fixed MC-GC network, find the odor representation by minimizing the 
    loss function using backtracking line search

    Inputs: 
    1) theta: gc threshold 
    2) odor_input: odor in R^50
    3) gc_init: initial value of gc activities (initial iterate)
    4) num_steps: total optimization steps
    5) gamma: line search parameter
    6) W: MC-GC network

    Outputs: 
    1) Net mc-activations (can be negative) for dictionary learning
    2) gc activations (non-negative)
    3) loss function at final iterate
    '''  
    loss_list = []
    iterate_list = [np.zeros(gc_init.size)]
    gc_act = gc_init.copy()
    
    for i in range(num_steps):
        net_mc = mc_input - np.matmul(np.transpose(W), gc_act)
        loss_before_step = get_loss(gc_act, net_mc, theta)       
        gradient = get_gradient(gc_act, net_mc, theta, W)
        norm_grad = gradient/sc.linalg.norm(gradient,2)
        
        # backtracking line search for projected gradient descent
        steplength = line_search_PGD(gc_act, net_mc, theta, W, norm_grad, loss_before_step, gamma)
        
        if steplength > 0:
            gc_act -= steplength*generalized_grad(gc_act, norm_grad, theta, steplength)
            
        loss_after_step = get_loss(gc_act, net_mc, theta)
        
        ## Sanity checks:
        # Ensure that iterates are within the feasible set
        less = [act for act in gc_act if 0.00000001 < act < theta] # is this just making sure we are 
        count_infeas = np.count_nonzero(less)
        if count_infeas > 0:  
            print('Step', i)
            print('Infeasibility after PGD step')
            print(less)
            break
            
        # Ensure that the loss monotonically decreases (provable convergence)
        if loss_after_step > loss_before_step:
            print('Something wrong at step', i, 'with steplength', steplength)
            print('Loss after', loss_after_step)
            print('Loss before', loss_before_step)
            break
            
        # stopping criteria
        if abs(loss_after_step - loss_before_step) < 0.01 and sc.linalg.norm(gc_act - iterate_list[i-1], 2) < 0.1:
            
            new_val = copy.deepcopy(net_mc)
            new_val[new_val <= thresh] = 0
            print('Converged at step', i, 'with steplength', steplength)
            print('Number of active MCs', np.count_nonzero(new_val))
            print('Number of active GCs', np.count_nonzero(gc_act))
            print('Function value at end', loss_after_step)
            break

        loss_list.append(loss_after_step)
        iterate_list.append(gc_act.copy())

    return net_mc, gc_act, loss_after_step