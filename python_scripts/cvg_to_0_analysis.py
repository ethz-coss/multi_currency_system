from Model import Model
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import pandas as pd
import sys
import csv
from os import path
 #TODO: parallelize
import multiprocessing
from itertools import product



"""
This script runs the model for different s/G ratios for stable systems, to study the impact of the 
increase constants on the equilibrium level of global parameters.

The output is stored in "equilibrium_vs_s_c_manyS.csv" and visualized in "eq_analysis.ipynb"

"""

# Number of iterations the model should run
runs = 3000
nreps = 20

init_global_params = 1000.0
global_threashold = 2000.0
token_threashold = 10.0
token_accounts = 0.0

#agentType = AgentType.DEFAULT

#const. total increase of global quantities
nagents = 100
nglob = 5
m = 1.0 
s = 1.0 #0.6 # s/G = 0.12
c_tot_vals = [0.1, 0.4, 0.7]
s_vals = [0.4, 0.8, 1.2]

# c_tot_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
# s_vals = c_tot_vals



def run_model(rng, c, m, s):
    # nagents = rng.integers(10, 500)
    # nglob = rng.integers(1, 10)

    increases = rng.uniform(1.0,10.0, nglob)
    increases = increases * c / np.sum(increases) #to ensure stabilizability

    #print(nagents, nglob)
    #increases = np.full(nglob, c / nglob)


    model_params = {"nr_agents": nagents, "initial_tokens": np.zeros(nglob), "nr_global_params": nglob, 
                            "mining_amounts": np.full(nglob, m), 
                        "spending_amount": s,
                        "global_quantities": np.full(nglob, init_global_params), 
                    "quantities_increase": increases, #np.random.uniform(0.05, 0.35, nglob), #np.full(nglob, 0.2), #  g_i += c_i after each step
                    "quantity_threashold": np.full((nagents,nglob), global_threashold),
                    "currency_threshold": np.full((nagents,nglob), token_threashold),
                        "token_accounts": np.full((nagents,nglob), token_accounts)}

  

    model  = Model(model_params["nr_agents"], 
                    model_params["nr_global_params"], 
                    model_params["quantity_threashold"], 
                    model_params["currency_threshold"], 
                    model_params["token_accounts"], 
                    model_params["global_quantities"], 
                    model_params["quantities_increase"], 
                    model_params["mining_amounts"],
                    model_params["spending_amount"], rng)

    model_data = model.run(runs)

    model_data['c_tot'] = c
    model_data['m'] = m
    model_data['s'] = s
    model_data['G'] = nglob
    model_data['step'] = np.arange(0,runs)
    model_data['c1, ..., cG'] = str(increases)
    #model_data['agent threasholds'] = str(increases)

    return model_data

    
def call_model(seed): # (seed, c, m)
    rng = np.random.default_rng(seed)
    data = []
    print('start', seed)
    for c_tot in c_tot_vals:
        for s in s_vals:
            print('seed, c, m: ', seed, c_tot, m)
            model_data = pd.DataFrame(run_model(rng, c_tot, m, s))
            model_data['r'] = seed
            data.append(model_data)
    
    print('end', seed)
    
    return pd.concat(data)

       


def main():
    
    # creating a pool
    p = multiprocessing.Pool() 
    # map list to target function 
    result = p.map(call_model, np.arange(nreps)) 
    pd.concat(result).to_csv("equilibrium_vs_s_c_manyS.csv")

    print("finished")

    
if __name__ == "__main__":
    main()