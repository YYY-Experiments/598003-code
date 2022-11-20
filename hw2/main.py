import numpy as np
import torch as tc
from utils import generate_GA , generate_T2 , generate_T1, rade_sketch_1 , rade_sketch_2 , leverage_score_sketch
from metrics import sketch_and_solve, iteration_complexity, condition_number
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm

n = 500
d = 50
trial_num = 100
ss = list(range(2*d, 5*d)) # range of s

def get_metric(metric, approxer, A, s, **kwargs):
    '''do multiple trials and take average'''
    results = []
    for trial_idx in range(trial_num):
        S = approxer(A , s)
        res = float( metric(A , S , **kwargs) )
        try:
            assert  res == res and res > -1e9 and res < 1e9  # detect nan
        except AssertionError:
            pdb.set_trace() # find nan

        results.append(res)
    results = tc.FloatTensor(results)
    return results.mean() , results.std()

def main(precondition, figname):
    plt.figure( figsize = (16 , 16) , dpi = 128 )
    plt.style.use('ggplot')

    for gene_idx , (generator , generator_name) in enumerate( zip( [generate_GA , generate_T2 , generate_T1] , ["GA" , "T2" , "T1"]) ):
        for metric_idx , (metric , metric_name) in enumerate( zip( 
            [sketch_and_solve, iteration_complexity, condition_number] , 
            ["Sketch and Solve", "Iteration Complexity", "Condition Number"]
        )):
            print (gene_idx , metric_idx)

            sb = None # ensure name
            if not precondition:
                if metric_idx == 0: # skip Sketch and Solve
                    continue
                sb = plt.subplot( 3,2 , gene_idx * 2 + metric_idx)
            else:
                sb = plt.subplot( 3,3 , gene_idx * 3 + metric_idx + 1 )

            sb.set_title("%s + %s" % (generator_name , metric_name))
            A = generator(n , d)
            for approx_idx , (approxer , approxer_name) in enumerate( zip(
                [ rade_sketch_1 , rade_sketch_2 , leverage_score_sketch ] , 
                [ "Sparse Rademacher sketch" , "Sparser Rademacher sketch" , "Leverage Score Sampling"]
            )):
                mean_results = []
                std_results = []
                for s in tqdm( ss ):
                    if metric_name in["Iteration Complexity" , "Condition Number"]:
                        mean, std = get_metric(metric, approxer , A, s, precondition = precondition)
                    else:
                        mean, std = get_metric(metric, approxer , A, s)
                    mean_results.append(mean)
                    std_results .append(std )

                plt.plot( ss , mean_results , label = approxer_name)                    
                # plt.fill_between( x , means[:,1] - std[:,1] , means[:,1] + std[:,1])

            plt.xlabel("s")
            plt.ylabel(metric_name)
            plt.legend()

    plt.tight_layout()
    plt.savefig( figname )
                
if __name__ == "__main__":
    main(False, "result_without_precond")
    main(True, "result_with_precond")
