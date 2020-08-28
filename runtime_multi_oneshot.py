from core.ttp import TTPSolver, TTPInstance
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import time
import pandas as pd
import seaborn as sns
"""
Test file fot TTP
"""


TTP_inst = TTPInstance("instances/ttp/ttp-a280-n279.txt")
TTP_solver = TTPSolver(TTP_inst)

def create_archive(solver,n=300):
    sols = []
    for i in range(n):
        sol = solver.independent_solver(tsp_init='random',tsp_local_search = "None",
                                        kp_init='random_thumwanit',kp_local_search="")
        sols.append((i,sol,sol.profit,np.ceil(sol.duration).astype(int)))
    return np.array(sols)

def one_shot_dominated(archive):
    #algorithm found in https://engineering.purdue.edu/~sudhoff/ee630/Lecture09.pdf
    sorted_archive = archive[np.argsort(-ttp_archive[:, 2])]
    P = list(sorted_archive[:,[0,3]])
    def Front(P):
        n = len(P)
        if n == 1:
            return P
        T = Front(P[:n//2])
        B = Front(P[n//2:])
        M = T.copy()
        for B_i in B:
            dominating =  all(B_i[1] < T_i[1]  for T_i in T)
            if dominating == True:
                M.append(B_i)
        return M
    dominating_sol = Front(P)
    dom_ind = list(np.array(dominating_sol)[:,0])
    return archive[dom_ind]

num_shots = [100,200,300,400,500]

time_usage = []

# Warm up
ttp_archive = create_archive(TTP_solver, 300)
one_shot_dom = one_shot_dominated(ttp_archive)

for num_shot in num_shots:
    time_usage_shot = []
    for i in range(20):
        ttp_archive = create_archive(TTP_solver, num_shot)

        st = time.time()
        one_shot_dom = one_shot_dominated(ttp_archive)
        ed = time.time()

        time_usage_shot.append(ed - st)

    time_usage.append(time_usage_shot)

df = pd.DataFrame(
    {key: arr for key,arr in zip(num_shots, time_usage)}
)
print(df)

sns.boxplot(data=df)
plt.xlabel('points')
plt.ylabel('exec. time (s)')
plt.savefig(f"experiment_data/runtime_analysis/oneshot_multi.png")
