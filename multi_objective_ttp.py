from core.ttp import TTPSolver, TTPInstance
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import time
import seaborn as sns
"""
Test file fot TTP
"""


TTP_inst = TTPInstance("instances/ttp/ttp-a280-n2790.txt")
TTP_solver = TTPSolver(TTP_inst)

def create_archive(solver,n=300):
    sols = []
    for i in range(n):
        sol = solver.independent_solver(tsp_init='random',tsp_local_search = "None",
                                        kp_init='random_thumwanit',kp_local_search="")
        sols.append((i,sol,sol.profit,np.ceil(sol.duration).astype(int)))
    return np.array(sols)

#ttp_archive = create_archive(TTP_solver)

def one_shot_dominated(archive):
    #algorithm found in https://engineering.purdue.edu/~sudhoff/ee630/Lecture09.pdf
    sorted_archive = archive[np.argsort(-archive[:, 2])]
    P = list(sorted_archive[:,[0,3]])
    def Front(P):
        n = len(P)
        if n == 1:
            return P
        T = Front(P[:n//2])
        B = Front(P[n//2:])
        M = T.copy()
        for B_i in B:
            print()
            dominating =  all(B_i[1] < T_i[1]  for T_i in T)
            if dominating == True:
                M.append(B_i)
        return M
    dominating_sol = Front(P)
    dom_ind = list(np.array(dominating_sol)[:,0])
    return archive[dom_ind]

#random_one_shot_dom = one_shot_dominated(ttp_archive)

def plot_pareto(archive,label=''):
    profits = archive[:,2]
    durations = archive[:,3]

    plt.scatter(durations,profits,label=label)
    plt.xlabel("Duration")
    plt.ylabel("Profit")
#plot_pareto(ttp_archive)
#plot_pareto(one_shot_dom)

################################################################

def scalarizing_moop(instance, C=[1,10,100]):
    sols = []
    exec_time = []
    for i in range(len(C)):
        start_time = time.time()
        instance.set_renting_ratio(C[i])
        TTP_solver = TTPSolver(instance)
        sol = TTP_solver.hyprid_EA(initial_strategy='greedy',
                                        n_population=10,n_offspring= 10,
                                        local_search=True,verbose=False)
        print(i)
        sols.append((i,sol,sol.profit,np.ceil(sol.duration).astype(int)))
        exec_time.append(time.time()-start_time)
    return np.array(sols), exec_time

C= np.logspace(-1,4,50)
ttp_scalarizing_moop,scalar_time = scalarizing_moop(TTP_inst,C)
scalar_dom = one_shot_dominated(ttp_scalarizing_moop)

#plot_pareto(ttp_scalarizing_moop,'Dominated')
#plot_pareto(scalar_dom,'Non-dominated')
#plt.legend()

#########################################3
def pareto_moop(instance,current_solutions = [], set_size = 1,n_steps = 20):
    
    def local_search_bitflip_moop(solution):
        
        P,T = TTP_solver.bitflip_neighbors_moop(solution)
        TSP_sol = solution.tsp_sol
        KP_sol = solution.kp_sol
        sols = []
        for i in range(len(P)-1): 
            KP_i = KP_sol.copy()
            KP_i[i] = (KP_i[i]+1)%2
            sols.append((i,KP_i,P[i],np.ceil(T[i]).astype(int)))
        dom_sols = one_shot_dominated(np.array(sols))
        for sol in dom_sols:
            sol[1] = instance.create_solution(TSP_sol, sol[1])
        
        return dom_sols
    
    if current_solutions == []:
        current_solutions = []
        for i in range(set_size):
            sol =TTP_solver.Heuristic_abdullah(start_ind=i)
            sol = TTP_solver.local_search(initial_solution = sol, neighborhood = '2-opt',verbose=False)
            current_solutions.append((i,sol,sol.profit,np.ceil(sol.duration).astype(int)))
    
    current_solutions = np.array(current_solutions)
    visited_solutions = []
    
    for i in range(n_steps):
        current_solutions[:,0] = np.arange(len(current_solutions))
        current_solutions = one_shot_dominated(current_solutions)
        for j in range(set_size):
            ind = np.random.randint(len(current_solutions))
            sol = current_solutions[ind,:]
            visited_solutions.append(sol)
            current_solutions = np.delete(current_solutions, ind, 0)
            new_sols = local_search_bitflip_moop(sol[1])
            current_solutions = np.concatenate((current_solutions,new_sols),axis=0)
            
    all_solutions = np.concatenate((current_solutions,np.array(visited_solutions)),axis=0)
    all_solutions[:,0] = np.arange(len(all_solutions))
    return one_shot_dominated(all_solutions)

set_size = [1]
n_steps = [1,10,50,100]
pairs = []
for i in set_size:
    for j in n_steps:
        pairs.append((i,j))

print("pareto local search")
exec_time = []
plt.figure(figsize=(8,6))
TTP_PARETO_MOOP = []
for pair in pairs:
    start_time = time.time()
    TTP_pareto_moop = pareto_moop(TTP_inst,set_size = 1, n_steps=1)
    for i in range(len(scalar_dom)):
        #TTP_inst.set_renting_ratio(c)
        #ttp_pareto_moop = pareto_moop(TTP_inst,set_size = 1, n_steps=1000)
        ttp_pareto_moop = pareto_moop(TTP_inst,[scalar_dom[i,:]],set_size = pair[0], n_steps=pair[1])
        #plot_pareto(ttp_pareto_moop,label=str(C[i]))
        TTP_pareto_moop = np.concatenate((TTP_pareto_moop,ttp_pareto_moop),axis=0)
        
    TTP_pareto_moop[:,0] = np.arange(len(TTP_pareto_moop))
    TTP_pareto_moop = one_shot_dominated(TTP_pareto_moop)
    exec_time.append(time.time()-start_time)
    TTP_PARETO_MOOP.append(TTP_pareto_moop)
    plot_pareto(TTP_pareto_moop,label='set_size: ' + str(pair[0]) + ' --n_steps: ' + str(pair[1]))
    
plot_pareto(TTP_pareto_moop,label='pareto_moop')
plot_pareto(scalar_dom,label='scalar_moop')

#plot_pareto(random_one_shot_dom,label='random')

plt.legend()

f, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,4))
ax1.boxplot(scalar_time)
ax1.set_title("Scalarizing MOOP with EA")
ax1.set_ylabel("Time")
ax1.set_xticks([])

sns.heatmap(np.array(exec_time).reshape((-1,4)),annot=True,fmt=".0f",yticklabels=set_size,xticklabels=n_steps,ax=ax2)
bottom, top = ax2.get_ylim()
ax2.set_ylim(bottom + 0.5, top - 0.5)
ax2.set_title("Pareto Local Search")


def reformat_sols(sols):
    all_sols = sols[0]
    for i in range(1,len(sols)):
        all_sols = np.concatenate((all_sols,sols[i]),axis=0)
    all_sols[:,0] = np.arange(len(all_sols))
    all_sols = one_shot_dominated(all_sols)
    
    reform = {'KP':[],'TSP':[],'Profit':[],'Duration':[]}
    for i in range(len(all_sols)):
        reform['KP'].append(list(all_sols[i,1].kp_sol.astype(int)))
        reform['TSP'].append(list(all_sols[i,1].tsp_sol))
        reform['Profit'].append(all_sols[i,2])
        reform['Duration'].append(all_sols[i,3])
    return reform

all_sols = reformat_sols(TTP_PARETO_MOOP)


with open('ttp_moop-a280-n2790.txt', 'w') as f:
    for key in ['Profit','Duration','KP','TSP']:
        f.write(key)
        f.write('\n' + "#"*20 + '\n')
        my_list = all_sols[key]
        for i in range(len(my_list)):
            row = my_list[i]
            try:
                for _string in row:
                    f.write(str(_string) + ' ')
                f.write('\n')
            except TypeError:
                continue
        
        f.write('\n')
        