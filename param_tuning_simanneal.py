from core.kp import KPSolver
from core.tsp import TSPSolver
from core.ttp import TTPSolver,TTPInstance

from progressbar import progressbar
import numpy as np
import matplotlib.pyplot as plt
import time
from glob import glob
import seaborn as sns
import pandas as pd


def sim_anneal_paramters_tuning(instance,T_max,T_ratio,n_cycles,n_steps_cycle,
                                problem = 'TTP',initial_strategy = 'greedy',neighborhood = 'bitflip',
                                verbose = False, repeat= 30):
    pairs = []
    for i in T_max:
        for j in T_ratio:
            for k in n_cycles:
                for l in n_steps_cycle:
                    pairs.append((i,j,k,l))

    strategies = {'KP':{'random':'random_thumwanit','greedy': 'density'},
                  'TSP':{'random':'random','greedy': 'to_nearest'},
                  'TTP':{'random':'random','greedy': 'greedy'}}

    data = list()
    for i in progressbar(range(repeat)):
        sub_data = list()
        for pair in pairs:
            start_time = time.time()
            EA_sol = instance.simulated_annealing(initial_strategy = strategies[problem][initial_strategy],neighborhood = neighborhood,
                                                  T_max = pair[0], T_min = pair[0]*pair[1],
                                                  n_cycles = pair[2], n_steps_cycle =pair[3], 
                                                  verbose=False)
            
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
    data = data.reshape((data.shape[0],-1))
    pairs = np.array(pairs)
    return data,pairs
    
def plot(data,pairs):
    df = pd.DataFrame({'T_max': pairs[:,0],'T_min': pairs[:,0]*pairs[:,1],
                       'N_cycles': pairs[:,2], 'N_steps_cycle': pairs[:,3],
                       'score': data[:,0], 'time': data[:,1]})
    #score
    f, axes = plt.subplots(3, 2,figsize=(15,12))
    
    sns.boxplot(x='T_max',y='score',data=df,ax=axes[0,0])
    sns.boxplot(x='T_min',y='score',data=df,ax=axes[0,1])
    sns.boxplot(x='N_cycles',y='score',data=df,ax=axes[1,0])
    sns.boxplot(x='N_steps_cycle',y='score',data=df,ax=axes[1,1])
    
    df_T= df.pivot_table(index='T_max',columns='T_min', values='score', aggfunc='mean')
    sns.heatmap(df_T,annot=True,fmt=".0f",ax=axes[2,0])
    bottom, top = axes[2,0].get_ylim()
    axes[2,0].set_ylim(bottom + 0.5, top - 0.5)
    
    df_C= df.pivot_table(index='N_cycles',columns='N_steps_cycle', values='score', aggfunc='mean')
    sns.heatmap(df_C,annot=True,fmt=".0f",ax = axes[2,1])
    bottom, top = axes[2,1].get_ylim()
    axes[2,1].set_ylim(bottom + 0.5, top - 0.5)
    
    #time  
    f2, axes2 = plt.subplots(3, 2,figsize=(15,12))
    sns.boxplot(x='T_max',y='time',data=df,ax=axes2[0,0])
    sns.boxplot(x='T_min',y='time',data=df,ax=axes2[0,1])
    sns.boxplot(x='N_cycles',y='time',data=df,ax=axes2[1,0])
    sns.boxplot(x='N_steps_cycle',y='time',data=df,ax=axes2[1,1])
    
    df_T= df.pivot_table(index='T_max',columns='T_min', values='time', aggfunc='mean')
    sns.heatmap(df_T,annot=True,fmt=".0f",ax=axes2[2,0])
    bottom, top = axes2[2,0].get_ylim()
    axes2[2,0].set_ylim(bottom + 0.5, top - 0.5)
    
    df_C= df.pivot_table(index='N_cycles',columns='N_steps_cycle', values='time', aggfunc='mean')
    sns.heatmap(df_C,annot=True,fmt=".0f",ax=axes2[2,1])
    bottom, top = axes2[2,1].get_ylim()
    axes2[2,1].set_ylim(bottom + 0.5, top - 0.5)

KP_instance = KPSolver("instances/kp/kp-n279.txt")
TSP_instance = TSPSolver("instances/tsp/tsp-a280.txt")
TTP_inst = TTPInstance("instances/ttp/ttp-a280-n279.txt")
TTP_instance = TTPSolver(TTP_inst)

T_max = [1,10,100]
T_ratio = [0.1,0.01,0.001]
n_cycles = [10,100,1000]
n_steps_cycle = [10,100,1000]
#data,pairs = sim_anneal_paramters_tuning(TTP_instance,T_max=T_max,T_ratio=T_ratio,
#                                   n_cycles = n_cycles,n_steps_cycle = n_steps_cycle,
#                                problem = 'TTP',initial_strategy = 'random',neighborhood = 'bitflip',
#                                verbose = True, repeat= 1)
plot(data,pairs)

