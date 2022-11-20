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

def main():
    plt.figure( figsize = (16 , 16) , dpi = 128 )
    plt.style.use('ggplot')

    for gene_idx , (generator , generator_name) in enumerate( zip( [generate_GA , generate_T2 , generate_T1] , ["GA" , "T2" , "T1"]) ):
        for metric_idx , (metric , metric_name) in enumerate( zip( 
            [sketch_and_solve, iteration_complexity, condition_number] , 
            ["Sketch and Solve", "Iteration Complexity", "Condition Number"]
        )):
            print (gene_idx , metric_idx)

            sb = plt.subplot( 3,3 , gene_idx * 3 + metric_idx + 1 )

            sb.set_title("%s + %s" % (generator_name , metric_name))
            A = generator(n , d)
            for approx_idx , (approxer , approxer_name) in enumerate( zip(
                [ rade_sketch_1 , rade_sketch_2 , leverage_score_sketch ] , 
                [ "Sparse Rademacher sketch $q=0.1$" , "Sparse Rademacher sketch $q=0.01$" , "Leverage Score Sampling"]
            )):
                mean_results = []
                std_results = []
                for s in tqdm( ss ):
                    mean, std = get_metric(metric, approxer , A, s)
                    mean_results.append(mean)
                    std_results .append(std )

                plt.plot( ss , mean_results , label = approxer_name)                    
                # plt.fill_between( x , means[:,1] - std[:,1] , means[:,1] + std[:,1])
            
            if metric_name in["Iteration Complexity" , "Condition Number"]:
                S = approxer(A , 1)
                ref_res = metric(A, S, precondition = False) # a reference without precondition
                plt.plot( ss , [ref_res for _ in ss] , label = "w/o pre-conditioning")         


            plt.xlabel("s")
            plt.ylabel(metric_name)
            plt.legend()

    plt.tight_layout()
    plt.savefig( "result" )
                
if __name__ == "__main__":
    main()