import numpy as nc 
import torch as tc
import scipy
import pdb
import random

# ---------- generators ----------
def generate_Sigma(d):
    _S = tc.arange(d)
    _S = (_S.view(1,d) - _S.view(d,1)).abs() # _S[i,j] = |i-j|
    Sigma = 2 * 0.5 ** _S # Sigma[i,j] = 2 * 0.5^{|i-j|}
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

# ---------- approximators ----------

def squared_row_sampling(A , s):
    n , d = A.size()
    p = (A ** 2).sum(-1) / (A ** 2).sum() # p[i]: the probability of row i

    # assert float( (p.sum() - 1).abs() ) < 1e-4

    p = [float(x.abs()) for x in p]
    p = [x / sum(p) for x in p]
    sampled = tc.LongTensor( nc.random.choice( list(range(n)) , s , replace = False , p = p) )

    tilde_A = A[sampled]
    return tilde_A.t() @ tilde_A

def leverage_score_sampling(A , s):
    n , d = A.size()

    r = tc.linalg.matrix_rank(A)
    AInv = tc.linalg.pinv(A.t() @ A)

    p = (A @ AInv @ A.t()).diag() / r
    # assert float( (p.sum() - 1).abs() ) < 1e-4

    p = [float(x.abs()) for x in p]
    p = [x / sum(p) for x in p]
    sampled = tc.LongTensor( nc.random.choice( list(range(n)) , s , replace = False , p = p) )

    tilde_A = A[sampled]
    return tilde_A.t() @ tilde_A

def gaussian_sketch(A , s):
    n , d = A.size()
    S = tc.randn(s , n) * (s ** -0.5)
    tilde_A = S @ A
    return tilde_A.t() @ tilde_A

def rade_sketch(A , s):
    n , d = A.size()
    S = ((tc.rand(s , n) < 0.5).float() * 2 - 1 ) * (s ** -0.5)
    tilde_A = S @ A
    return tilde_A.t() @ tilde_A

# ---------- metrics ----------

def Fnorm(P):
    return (P ** 2).sum() ** 0.5

def spectral_norm(P):
    try: 
        res = tc.linalg.eigvalsh(P + 1e-6).abs().max()
    except tc._C._LinAlgError: 
        print ("fuck!")
        x = tc.randn(P.size(0))
        res = float( (x.t() @ P @ x) / (Fnorm(x)**2) )
    return res

def metric_F(AA , TAA):
    '''TAA: approximated A.t() @ A'''
    return Fnorm(TAA - AA) / Fnorm(AA)

def metric_spectral(AA , TAA):
    return spectral_norm(TAA - AA) / spectral_norm(AA)

def metric_spectral_approx(AA , TAA):
    d = AA.size(0)
    L , U = tc.linalg.eigh(AA) # AA = U @ L.diag() @ U.t()
    AAhalfinv = U @ ((L + 1e-6) ** -0.5).diag() @ U.t()
    return spectral_norm(AAhalfinv @ TAA @ AAhalfinv - tc.eye(d,d))