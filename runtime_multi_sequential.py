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


def sequential_non_dominated(sol, sorted_archive):
    if len(sorted_archive) == 0:
        sorted_archive.append([sol])
    else:
        for i, sol_list in enumerate(sorted_archive):
            larger_prof = False
            inferior_idx = None
            for idx, pl_sol in enumerate(sol_list):
                if pl_sol[0] > sol[0]:
                    larger_prof = True
                    break
                else:
                    if pl_sol[1] > sol[1] and inferior_idx is None:
                        inferior_idx = idx

            if larger_prof:
                if sol_list[idx][1] <= sol[1]:
                    continue
                else:
                    sol_list.insert(idx, [sol[0], sol[1]])
                    if inferior_idx is not None:
                        inferior_points = sol_list[inferior_idx: idx].copy()
                        del sol_list[inferior_idx:idx]

                        # Find place for inferior points
                        for inf_point in inferior_points:
                            sequential_non_dominated(inf_point, sorted_archive[i:])
                    return

            else:
                sol_list.append([sol[0], sol[1]])
                if inferior_idx is not None:
                    inferior_points = sol_list[inferior_idx: idx+1].copy()
                    del sol_list[inferior_idx:idx+1]

                    # Find place for inferior points
                    for inf_point in inferior_points:
                        sequential_non_dominated(inf_point, sorted_archive[i:])
                return

        sorted_archive.append([[sol[0], sol[1]]]) 


def create_sol(solver, n=300):
    sols = []
    for i in range(n):
        sol = solver.independent_solver(tsp_init='random',tsp_local_search = "None",
                                        kp_init='random_thumwanit',kp_local_search="")
        sols.append([sol.profit,np.ceil(sol.duration)])
    return sols

def create_archive(sols):
    sorted_archive = []
    for sol in sols:
        sequential_non_dominated(sol, sorted_archive=sorted_archive)

num_shots = [100,200,300,400,500]

time_usage = []

# Warm up
num_shot = 500
sols = create_sol(TTP_solver, num_shot)
create_archive(sols)

for num_shot in num_shots:
    time_usage_shot = []
    for i in range(20):
        sols = create_sol(TTP_solver, num_shot)

        st = time.time()
        create_archive(sols)
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
plt.savefig(f"experiment_data/runtime_analysis/sequence_multi.png")

    