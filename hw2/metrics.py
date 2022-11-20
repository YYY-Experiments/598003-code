import numpy as np
import pdb
import torch as tc
from scipy.sparse.linalg import lsqr as lsqr

def L(A,b,x):
    return ((A @ x - b)**2).sum()

def get_error_rate(A,b,x_tilde):
    x_star = tc.linalg.inv(A.t() @ A) @ A.t() @ b

    return  (L(A,b,x_tilde) - L(A,b,x_star)) / ( L(A,b,x_star) + 1e-9)

def get_b(A):
    n, d = A.size()
    x_bar = tc.randn(d) / (d**0.5)
    xi = tc.randn(n) / (d**0.5)
    return A @ x_bar + xi

def get_x_0_tilde(A_tilde,b_tilde):
    x_0_tilde = tc.linalg.inv(A_tilde.t() @ A_tilde) @ A_tilde.t() @ b_tilde
    return x_0_tilde


def sketch_and_solve(A, S):
    n, d = A.size()
    b = get_b(A)


    A_tilde = S @ A
    b_tilde = S @ b

    x_0_tilde = get_x_0_tilde(A_tilde , b_tilde)
    
    return get_error_rate(A,b,x_0_tilde)



def iteration_complexity(A,S, precondition):
    n, d = A.size()
    b = get_b(A)

    A_tilde = S @ A
    b_tilde = S @ b
    x_0_tilde = get_x_0_tilde(A_tilde , b_tilde) # starting point

    x0 = tc.zeros(d)
    the_A = A
    Rinv = None # ensure the name exists
    if precondition:
        R = tc.linalg.qr(A_tilde, mode = "r").R
        Rinv = tc.linalg.pinv(R)
        the_A = A @ Rinv
        x0 = x_0_tilde
    ret = lsqr(the_A.numpy(), b.numpy(), iter_lim = 1e8, x0 = x0)
    iteration_num = ret[2]

    x_tilde = Rinv @ ret[0] if precondition else ret[0]
    try:
        assert get_error_rate(A,b,x_tilde) < 1e-5 # ensure error rate < 1e-5
    except AssertionError:
        pdb.set_trace()
    
    return iteration_num
    
def condition_number(A,S,precondition):
    n, d = A.size()
    b = get_b(A)

    the_A = A
    A_tilde = S @ A
    if precondition:
        R = tc.linalg.qr(A_tilde, mode = "r").R
        Rinv = tc.linalg.pinv(R)
        the_A = A @ Rinv

    return tc.linalg.cond(the_A)
