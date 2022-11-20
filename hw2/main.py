import numpy as np
import torch as tc
from utils import generate_GA , generate_T2 , generate_T1 , metric_F , metric_spectral , metric_spectral_approx , squared_row_sampling , leverage_score_sampling , gaussian_sketch , rade_sketch
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm

if __name__ == "__main__":
    plt.figure( figsize = (16 , 16) , dpi = 128 )
    plt.style.use('ggplot')

    n = 500
    d = 50
    trial_num = 100
    ss = list(range(d // 2 , 4 * d)) # range of s


    for gene_idx , (generator , generator_name) in enumerate( zip( [generate_GA , generate_T2 , generate_T1] , ["GA" , "T2" , "T1"]) ):
        for metric_idx , (metric , metric_name) in enumerate( zip( [metric_F , metric_spectral , metric_spectral_approx] , ["Frobenius norm" , "spectral norm" , "spectral approximation"])):
            print (gene_idx , metric_idx)
            sb = plt.subplot( 3,3 , gene_idx * 3 + metric_idx + 1 )
            sb.set_title("%s + %s" % (generator_name , metric_name))
            results_tot = [[[] for  _ in range(trial_num)] for __ in range(4)]
            for trial_idx in tqdm( range(trial_num) ):
                A = generator(n , d)
                AA = A.t() @ A
                for approx_idx , (approxer , approxer_name) in enumerate( zip(
                    [ squared_row_sampling , leverage_score_sampling , gaussian_sketch , rade_sketch ] , 
                    [ "Squared row norm sampling" , "Leverage score sampling" , "Gaussian sketch" , "Rademacher sketch"]
                )):
                    last_res = 0 # last time data, for data missing
                    for s in ss:
                        TAA = approxer(A , s)
                        res = float( metric(AA , TAA) )
                        if res == res and res > -1e9 and res < 1e9: # avoid nan
                            results_tot[approx_idx][trial_idx].append( [s , res ] ) # results[approx_idx][i,j,k]: ith trial, s = j, result = k
                            last_res = res
                        else:
                            results_tot[approx_idx][trial_idx].append( [s , last_res ] ) # use last time data
            for approx_idx , approxer_name in enumerate([ "Squared row norm sampling" , "Leverage score sampling" , "Gaussian sketch" , "Rademacher sketch"]):
                results = tc.Tensor(results_tot[approx_idx])
                means = results.mean(0)
                #std = results.std(0)

                x = tc.LongTensor(ss)
                plt.plot( x , means[:,1] , label = approxer_name)
                # plt.fill_between( x , means[:,1] - std[:,1] , means[:,1] + std[:,1])
            plt.xlabel("s")
            plt.ylabel(metric_name)
            plt.legend()
    # plt.show()
    plt.tight_layout()
    plt.savefig( "result" )
                

