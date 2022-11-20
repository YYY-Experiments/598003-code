import numpy as nc 
import torch as tc
import scipy
import pdb
import random

# ---------- generators ----------
def generate_Sigma(d):
    _S = tc.arange(d)
    _S = (_S.view(1,d) - _S.view(d,1)).abs() # _S[i,j] = |i-j|
    Sigma = 2 * 0.9 ** _S # Sigma[i,j] = 2 * 0.5^{|i-j|}
    return Sigma


def generate_GA(n , d):
    '''return: (n,d)'''
    return tc.Tensor( nc.random.multivariate_normal(tc.ones(d) , generate_Sigma(d) , n) )


def generate_T2(n , d):
    '''return: (n,d)'''
    return tc.Tensor(scipy.stats.multivariate_t(tc.zeros(d) , generate_Sigma(d), 2).rvs(size = n))

def generate_T1(n , d):
    '''return: (n,d)'''
    return tc.Tensor(scipy.stats.multivariate_t(tc.zeros(d) , generate_Sigma(d), 1).rvs(size = n))

# ---------- sketching matrix ----------

def rade_sketch_1(A , s):
    n , d = A.size()
    q = 0.1
    
    B = (tc.rand(s , n) < q).float() # bernuli
    R = ((tc.rand(s , n) < 0.5).float() * 2 - 1 ) # rade
    S = B * R * ( (q*s) ** -0.5 ) 
    
    return S

def rade_sketch_2(A , s):
    n , d = A.size()
    q = 0.01
    
    B = (tc.rand(s , n) < q).float() # bernuli
    R = ((tc.rand(s , n) < 0.5).float() * 2 - 1 ) # rade
    S = B * R * ( (q*s) ** -0.5 ) 
    
    return S

def leverage_score_sketch(A , s):
    n , d = A.size()

    r = tc.linalg.matrix_rank(A)
    AInv = tc.linalg.pinv(A.t() @ A)

    p = (A @ AInv @ A.t()).diag() / r
    # assert float( (p.sum() - 1).abs() ) < 1e-4

    p = [float(x.abs()) for x in p]
    p = [x / sum(p) for x in p]
    sampled = tc.LongTensor( nc.random.choice( list(range(n)) , s , replace = False , p = p) )
    S = tc.zeros(s,n)
    S[tc.arange(s) , sampled] = 1

    return S

