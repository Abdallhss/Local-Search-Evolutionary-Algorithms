from core.tsp import TSPSolver
from glob import glob
from time import time
from progressbar import progressbar
import numpy as np
import matplotlib.pyplot as plt


instances_uris = glob("instances/tsp/*")
# Comment next line to iterate on all instances
#instances_uris = ("instances/tsp/tsp-a280.txt",)


def try_strategies(instance, repeat=10):
    data = list()
    for i in progressbar(range(repeat)):
        sub_data = list()

        """
        # Random strategy
        start = time()
        solution = instance.solve(initial_strategy="random")
        distance = instance.evaluate(solution)
        sub_data.append((distance, time() - start))
        """

        # To nearest strategy
        start = time()
        solution = instance.solve(initial_strategy="to_nearest")
        distance = instance.evaluate(solution)
        sub_data.append((distance, time() - start))


        # To nearest with 2opt local search
        start = time()
        solution = instance.solve(initial_strategy="to_nearest", local_search_strategy="2opt")
        distance = instance.evaluate(solution)
        sub_data.append((distance, time() - start))

        # To nearest with hybride EA
        start = time()
        solution = instance.hyprid_EA(initial_strategy = 'to_nearest',n_population = 5,n_offspring = 5,verbose = False)
        distance = instance.evaluate(solution)
        sub_data.append((distance, time() - start))

        # To nearest with simulated annealing
        start = time()
        solution = instance.simulated_annealing(initial_strategy='to_nearest', neighborhood='2opt', verbose=False)
        distance = instance.evaluate(solution)
        sub_data.append((distance, time() - start))


        data.append(sub_data)

    # If you add a strategy, please add its label here:
    strategies_labels = ("to_nearest", "local_search", "hybrid_EA_5_5", "annealing")

    data = np.array(data).transpose(1, 0, 2)
    return strategies_labels, data


for instance_uri in instances_uris:
    try:
        instance = TSPSolver(instance_uri)
    except:
        continue
    name = instance_uri.split("/")[-1]
    # Get a array of size (n_strategies x n_repeat x 2)
    # The dimension of size 3 is for (length, time duration)
    # The dimension of size n_repeat is for average and standard deviation purpose




    strategies_labels, data = try_strategies(instance)

    plt.boxplot(data[:, :, 0].T)
    plt.xticks(range(1, len(strategies_labels) + 1), strategies_labels, rotation=45)
    plt.title("distance for: " + name)
    plt.savefig("./experiment_data/overall_analysis/tsp/" + name + ".distance.png", bbox_inches='tight')
    plt.clf()


    plt.boxplot(data[:, :, 1].T)
    plt.xticks(range(1, len(strategies_labels) + 1), strategies_labels, rotation=45)
    plt.title("time in seconds for: " + name)
    plt.savefig("./experiment_data/overall_analysis/tsp/" + name + ".time.png", bbox_inches='tight')
    plt.clf()