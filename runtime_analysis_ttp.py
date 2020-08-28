from core.ttp import TTPSolver, TTPInstance
from glob import glob
from time import time
from progressbar import progressbar
import numpy as np
import matplotlib.pyplot as plt

INSTANCE_PATH = "instances/ttp/"

instances_uris = glob(INSTANCE_PATH + "*")
# Comment next lines to iterate on all instances
instances = ["ttp-a280-n2790"]
instances_uris = (INSTANCE_PATH + name + ".txt" for name in instances)

def try_strategies(instance, repeat= 1):
    data = list()
    for i in progressbar(range(repeat)):
        sub_data = list()

        # bit flip with zero start
        greedy_sol = instance.all_random_solution()
        solution, score = instance.local_search(initial_solution=greedy_sol, stratergy='best',
             neighborhood='bitflip', dynamic_neighbor=True, log=True)
        sub_data.append(score)

        greedy_sol = instance.Heuristic_abdullah(tsp_init='to_nearest')
        solution, score = instance.local_search(initial_solution=greedy_sol, stratergy='best',
             neighborhood='bitflip', dynamic_neighbor=True, log=True)
        sub_data.append(score)

        #Hybride_EA with density heuristic
        # start = time()
        # solution, _, profit_list = instance.hyprid_EA(initial_strategy = 'random_low_values',
        #                                       n_population = 5,n_offspring = 5,verbose = False, log=True)
        # profit, weight, feasible = instance.evaluate(solution)
        # sub_data.append(profit_list)

        # start = time()
        # solution, weight = instance.random_valentin()
        # profit, weight, feasible = instance.evaluate(solution)
        # sub_data.append(( profit, weight, time() - start))

        # start = time()
        # solution, weight = instance.random_thumwanit()
        # profit, weight, feasible = instance.evaluate(solution)
        # sub_data.append((profit, weight, time() - start))

        # #density heuristic
        # start = time()
        # solution, weight = instance.density_heuristic()
        # profit, weight, feasible = instance.evaluate(solution)
        # sub_data.append((profit, weight, time() - start))

        # #density_heuristic + bitsflip
        # start = time()
        # solution, weight = instance.density_heuristic()
        # solution, weight = instance.bits_flip_local_search(solution, weight)
        # profit, weight, feasible = instance.evaluate(solution)
        # sub_data.append((profit, weight, time() - start))

        # #Hybride_EA with density heuristic
        # start = time()
        # solution, _ = instance.hyprid_EA(initial_strategy = 'density',
        #                                       n_population = 5,n_offspring = 5,verbose = False)
        # profit, weight, feasible = instance.evaluate(solution)
        # sub_data.append((profit, weight, time() - start))

        data.append(sub_data)

    # If you add a strategy, please add its label here:
    strategies_labels  = ("allrandom-nearest-bitflip-dynamic", "Abdullah-nearest-bitflip-dynamic",)#,"Hybride_EA/5/5")
    # strategies_labels = ("random_abdullah" , "random_valentin","random_thumwanit",
    #                      "density_heuristic", "density_heuristic/bitsflip_localsearch" ,"Hybride_EA/5/5")

    return strategies_labels, data

for instance_name, instance_uri in zip(instances, instances_uris):
    print(instance_uri)
    instance = TTPSolver(TTPInstance(instance_uri))
    # Get a array of size (n_strategies x n_repeat x 3)
    # The dimension of size 3 is for (profit, weight, time duration)
    # The dimension of size n_repeat is for average and standard deviation purpose

    strategies_labels, data = try_strategies(instance, repeat=1)

    for idx in range(len(strategies_labels)):
        label = strategies_labels[idx]
        iter_data = list()
        all_max = []
        for sub_data in data:
            # Normalize

            tmp = sub_data[idx].copy()
            tmp = (tmp - tmp[0]) / (tmp[-1] - tmp[0])
            iter_data.append(tmp)
            all_max.append(sub_data[idx][-1])
        
        longest_iter = max([len(x) for x in iter_data])
        profit_record = np.zeros(longest_iter)

        for i in range(longest_iter):
            profit_at_iter = list()
            for iter_record in iter_data:
                if len(iter_record) < (i+1): profit_at_iter.append(1.)
                else: profit_at_iter.append(iter_record[i])
            
            print(profit_at_iter)

            profit_record[i] = np.mean(profit_at_iter)

            
        # print(data)
        # print(profit_record)
        x_plot = np.arange(1, longest_iter+1)
        plt.plot(x_plot, profit_record)
        plt.text(0.1, 0.9, f"Maximum Score: {np.mean(all_max)}")
        plt.xlabel('Iteration')
        plt.ylabel('Relative optimality')
        plt.savefig(f"experiment_data/runtime_analysis/{instance_name}_{strategies_labels[idx]}.png")
        plt.cla()
        plt.clf()
