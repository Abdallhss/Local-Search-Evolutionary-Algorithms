from core.ttp import TTPSolver, TTPInstance
from glob import glob
from time import time
from progressbar import progressbar
import numpy as np
import matplotlib.pyplot as plt



instances_uris = glob("instances/ttp/*")
# Comment next line to iterate on all instances
#instances_uris = ("instances/ttp/ttp-a280-n1395.txt",)


def try_strategies(instance, repeat=10):
    data = list()
    for i in progressbar(range(repeat)):
        sub_data = list()

        # Independent solution
        start = time()
        solution = instance.independent_solver()
        sub_data.append((solution.score, time() - start))

        # Local search on greedy solution
        start = time()
        greedy_sol = instance.Heuristic_abdullah(tsp_init='to_nearest')
        solution = instance.local_search(greedy_sol, neighborhood='2-opt', verbose=False)
        sub_data.append((solution.score, time() - start))

        """
        # Grasp
        start = time()
        solution = instance.ILS_GRASP(neighborhood = '2-opt',verbose=False)
        sub_data.append((solution.score, time() - start))
        """

        # Simulated annealing
        start = time()
        solution = instance.simulated_annealing(neighborhood = 'bitflip',verbose=False)
        sub_data.append((solution.score, time() - start))


        """
        # hybrid EA 20 20
        start = time()
        solution = instance.hyprid_EA(n_population=20,n_offspring=20)
        sub_data.append((solution.score, time() - start))
        """


        data.append(sub_data)

    # If you add a strategy, please add its label here:
    strategies_labels = ("independent", "local_search", "annealing") #, "hybrid_EA/20/20")

    data = np.array(data).transpose(1, 0, 2)
    return strategies_labels, data


for instance_uri in instances_uris:
    try:
        instance = TTPSolver(TTPInstance(instance_uri))
    except:
        continue
    name = instance_uri.split("/")[-1]
    # Get a array of size (n_strategies x n_repeat x 2)
    # The dimension of size 3 is for (length, time duration)
    # The dimension of size n_repeat is for average and standard deviation purpose

    strategies_labels, data = try_strategies(instance)

    plt.boxplot(data[:, :, 0].T)
    plt.xticks(range(1, len(strategies_labels) + 1), strategies_labels, rotation=45)
    plt.title("score for: " + name)
    plt.savefig("./experiment_data/overall_analysis/ttp/" + name + ".score.png",  bbox_inches='tight')
    plt.clf()


    plt.boxplot(data[:, :, 1].T)
    plt.yscale("log")
    plt.xticks(range(1, len(strategies_labels) + 1), strategies_labels, rotation=45)
    plt.title("time in seconds for: " + name)
    plt.savefig("./experiment_data/overall_analysis/ttp/" + name + ".time.png",  bbox_inches='tight')
    plt.clf()