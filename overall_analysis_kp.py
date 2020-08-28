from core.kp import KPSolver
from glob import glob
from time import time
from progressbar import progressbar
import numpy as np
import matplotlib.pyplot as plt

instances_uris = glob("instances/kp/*")
# Comment next lines to iterate on all instances
#instances_uris = ("instances/kp/kp-n279.txt",)

def try_strategies(instance, repeat= 10):
    data = list()
    for i in progressbar(range(repeat)):
        sub_data = list()

        """
        #random strategies
        start = time()
        solution, _ = instance.random_abdullah()
        profit, weight, feasible = instance.evaluate(solution)
        sub_data.append(( profit, weight, time() - start))

        start = time()
        solution, weight = instance.random_valentin()
        profit, weight, feasible = instance.evaluate(solution)
        sub_data.append(( profit, weight, time() - start))

        start = time()
        solution, weight = instance.random_thumwanit()
        profit, weight, feasible = instance.evaluate(solution)
        sub_data.append((profit, weight, time() - start))
        """

        #density heuristic
        start = time()
        solution, weight = instance.density_heuristic()
        profit, weight, feasible = instance.evaluate(solution)
        sub_data.append((profit, weight, time() - start))

        #density_heuristic + bitsflip
        start = time()
        solution, weight = instance.density_heuristic()
        solution, weight = instance.bits_flip_local_search(solution, weight)
        profit, weight, feasible = instance.evaluate(solution)
        sub_data.append((profit, weight, time() - start))
        
        #Simulated Annealing + bitflip
        start = time()
        solution, profit = instance.simulated_annealing(initial_strategy = 'density',
                                                        neighborhood = 'swap',verbose=False)
        profit, weight, feasible = instance.evaluate(solution)
        sub_data.append((profit, weight, time() - start))
        
        
        #Hybride_EA with density heuristic
        start = time()
        solution, _ = instance.hyprid_EA(initial_strategy = 'density',
                                              n_population = 5,n_offspring = 5,verbose = False)
        profit, weight, feasible = instance.evaluate(solution)
        sub_data.append((profit, weight, time() - start))

        data.append(sub_data)

    # If you add a strategy, please add its label here:
    strategies_labels = (#"random_abdullah", "random_valentin","random_thumwanit",
                         "density_heuristic", "local_search" ,
                         "annealing_bitflip","hybride_EA/5/5")

    data = np.array(data).transpose(1,0,2)
    return strategies_labels, data

for instance_uri in instances_uris:
    print(instance_uri)
    try:
        instance = KPSolver(instance_uri)
    except:
        continue


    name = instance_uri.split("/")[-1]
    # Get a array of size (n_strategies x n_repeat x 3)
    # The dimension of size 3 is for (profit, weight, time duration)
    # The dimension of size n_repeat is for average and standard deviation purpose

    strategies_labels, data = try_strategies(instance)

    plt.boxplot(data[:,:,0].T)
    plt.xticks(range(1, len(strategies_labels[:])+1), strategies_labels[:], rotation=45)
    plt.title("profit for: " + name)
    plt.savefig("./experiment_data/overall_analysis/kp/" + name + ".profit.png",  bbox_inches='tight')
    plt.clf()

    plt.boxplot(data[:, :, 1].T)
    plt.xticks(range(1, len(strategies_labels) + 1), strategies_labels, rotation=45)
    plt.title("weight for: " + name)
    plt.savefig("./experiment_data/overall_analysis/kp/" + name + ".weight.png",  bbox_inches='tight')
    plt.clf()

    plt.boxplot(data[:,:,2].T)
    plt.xticks(range(1, len(strategies_labels)+1), strategies_labels, rotation=45)
    plt.title("time in seconds for: " + name)
    plt.savefig("./experiment_data/overall_analysis/kp/" + name + ".time.png", bbox_inches='tight')
    plt.clf()
