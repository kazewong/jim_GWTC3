import os
import numpy as np


#################### Directories Management ####################

def mkdir(path):
    """
    To create a directory if it does not exist
    """
    if not os.path.exists(path):
        os.makedirs(path)
        

#################### Miscellaneous Functions ####################
    
def KLdivergence(x, y):
    """
    Compute the Kullback-Leibler divergence between two samples.
    Parameters
    ----------
    x : 1D np array
        Samples from distribution P, which typically represents the true
        distribution.
    y : 1D np rray
        Samples from distribution Q, which typically represents the approximate
        distribution.
    Returns
    -------
    out : float
        The estimated Kullback-Leibler divergence D(P||Q).
    """ 
    bins = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)), 100) 
    # The size of array must be much greater than the number of bins
    
    prob_x = np.histogram(x, bins=bins)[0]/x.shape
    prob_y = np.histogram(y, bins=bins)[0]/y.shape
       
    return np.sum(np.where((prob_y != 0) & (prob_x != 0), prob_y * (np.log(prob_y) - np.log(prob_x)), 0))


def JSdivergence(P, Q):
    """
    Compute the Jensen-Shannon divergence between two samples.
    Parameters
    ----------
    P : 1D np array
        Samples from distribution P
    Q : 1D np array
        Samples from distribution Q
    Returns
    -------
    out : float
        The estimated Jensen-Shannon divergence D(P||Q).
    """
    M=(P+Q)/2 #sum the two distributions then get average

    kl_p_q = KLdivergence(P, M)
    kl_q_p = KLdivergence(Q, M)

    js = (kl_p_q+kl_q_p)/2
    return js
