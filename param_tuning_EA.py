from core.kp import KPSolver
from core.tsp import TSPSolver
from core.ttp import TTPSolver,TTPInstance

from progressbar import progressbar
import numpy as np
import matplotlib.pyplot as plt
import time
from glob import glob
import seaborn as sns


def EA_paramters_tuning(instance,n_pop,n_offspring,
                        problem = 'TSP',initial_strategy = 'greedy',
                        local_search=False,verbose = False, repeat= 30):
    pairs = []
    for i in n_pop:
        for j in n_offspring:
            pairs.append((i,j))
    strategies = {'KP':{'random':'random_thumwanit','greedy': 'density'},
                  'TSP':{'random':'random','greedy': 'to_nearest'},
                  'TTP':{'random':'random','greedy': 'greedy'}}

    data = list()
    for i in progressbar(range(repeat)):
        sub_data = list()
        for pair in pairs:
            start_time = time.time()
            #TSP
            EA_sol = instance.hyprid_EA(initial_strategy=strategies[problem][initial_strategy],
                                        n_population=pair[0],n_offspring=pair[1],
                                        local_search=local_search,verbose=False
                                        )
            if problem == 'KP':
                sol = instance.evaluate(EA_sol[0])[0]
            elif problem == 'TSP':
                sol = instance.evaluate(EA_sol)
            elif problem == 'TTP':
                sol = EA_sol.score
            if verbose:
                print(i,pair,sol)
            sub_data.append((sol,time.time() - start_time))
        data.append(sub_data)
    data = np.array(data).transpose(1,0,2)
    return data
    
def plot(data,n_pop,n_offspring):
    n = len(n_pop)
    o = len(n_offspring)
    data_mean = np.mean(data,axis=1)
    mean_dist = data_mean[:,0].reshape((n ,-1))
    mean_time = data_mean[:,1].reshape((n ,-1))
    data_std = np.std(data,axis=1)
    std_dist = data_std[:,0].reshape((n ,-1))
    std_time = data_std[:,1].reshape((n ,-1))
    
    f, axes = plt.subplots(3, 2,figsize=(15,12))
    'quality: '
    #plt.figure(figsize=(8,6))
    sns.heatmap(mean_dist,annot=True,fmt=".0f",yticklabels=n_pop,xticklabels=n_offspring,ax = axes[0,0])
    bottom, top = axes[0,0].get_ylim()
    axes[0,0].set_ylim(bottom + 0.5, top - 0.5)
    axes[0,0].set_title("Profit",fontsize=20)
    #plt.figure(figsize=(8,6))
    for i,n_o in enumerate(n_offspring):
        axes[1,0].errorbar(range(n),mean_dist[:,i],std_dist[:,i],label='No. offspring = '+ str(n_o))
        axes[1,0].set_xticks(range(n))
        axes[1,0].set_xticklabels(n_pop)
    axes[1,0].errorbar(range(n),np.mean(mean_dist,axis=1),np.mean(std_dist,axis=1),linestyle = '--', linewidth = 2,label='Mean')
    axes[1,0].legend()
    
    #plt.figure(figsize=(8,6))
    for i,n_p in enumerate(n_pop):
        axes[2,0].errorbar(range(o),mean_dist[i,:],std_dist[i,:],label='Pop size = '+ str(n_p))
        axes[2,0].set_xticks(range(o)) 
        axes[2,0].set_xticklabels(n_offspring)
    axes[2,0].errorbar(range(o),np.mean(mean_dist,axis=0),np.mean(std_dist,axis=0),linestyle = '--', linewidth = 2,label='Mean')
    axes[2,0].legend()
    
    'Time: '
    #plt.figure(figsize=(8,6))
    sns.heatmap(mean_time,annot=True,fmt=".0f",yticklabels=n_pop,xticklabels=n_offspring,ax=axes[0,1])
    bottom, top = axes[0,1].get_ylim()
    axes[0,1].set_ylim(bottom + 0.5, top - 0.5)
    axes[0,1].set_title("Time",fontsize=20)
    
    #plt.figure(figsize=(8,6))
    for i,n_o in enumerate(n_offspring):
        axes[1,1].errorbar(range(n),mean_time[:,i],std_time[:,i],label='No. offspring = '+ str(n_o))
        axes[1,1].set_xticks(range(n))
        axes[1,1].set_xticklabels(n_pop)
    axes[1,1].errorbar(range(n),np.mean(mean_time,axis=1),np.mean(std_time,axis=1),linestyle = '--', linewidth = 2,label='Mean')
    axes[1,1].legend()
    
    #plt.figure(figsize=(8,6))
    for i,n_p in enumerate(n_pop):
        axes[2,1].errorbar(range(o),mean_time[i,:],std_time[i,:],label='Pop size = '+ str(n_p))
        axes[2,1].set_xticks(range(o))
        axes[2,1].set_xticklabels(n_offspring)
    axes[2,1].errorbar(range(o),np.mean(mean_time,axis=0),np.mean(std_time,axis=0),linestyle = '--', linewidth = 2,label='Mean')
    axes[2,1].legend()



KP_instance = KPSolver("instances/kp/kp-n279.txt")
TSP_instance = TSPSolver("instances/tsp/tsp-a280.txt")
TTP_inst = TTPInstance("instances/ttp/ttp-a280-n279.txt")
TTP_instance = TTPSolver(TTP_inst)

n_pop = [5,10,20,50,100]
n_offspring = [10,20,50,100,200]

#data = EA_paramters_tuning(TTP_instance,n_pop=n_pop,n_offspring=n_offspring,problem = 'TTP',initial_strategy = 'random',local_search=True,verbose = True, repeat= 10)
plot(data,n_pop,n_offspring)

