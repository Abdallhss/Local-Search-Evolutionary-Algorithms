from core.ttp import TTPSolver, TTPInstance
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
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

sorted_archive = []

def create_archive(solver,n=300):
    sols = []
    for i in range(n):
        sol = solver.independent_solver(tsp_init='random',tsp_local_search = "None",
                                        kp_init='random_thumwanit',kp_local_search="")
        sequential_non_dominated([sol.profit, np.ceil(sol.duration)], sorted_archive=sorted_archive)

num_shot = 500
create_archive(TTP_solver, 500)

plt.xlabel("Duration")
plt.ylabel("Profit")

for sol_list in sorted_archive:
    to_plot = np.array(sol_list)
    plt.plot(to_plot[:,1], to_plot[:,0])

plt.savefig(f"experiment_data/pareto/sequential_{num_shot}.png")

    