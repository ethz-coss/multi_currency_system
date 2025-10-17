from Model import Model
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import pandas as pd
import sys
import csv
import multiprocessing



"""
This script runs a stable system where a sudden jump of one global parameters occurs with different sizes, 
to study the recovery time after critical events.

Output is stored in the file "recovery_time_linear_bigS.csv" and visualized in the file "crisis_figures.ipynb"

"""

seed_glob = 108
rng_glod = np.random.default_rng(seed_glob)
numReps = 50


# Number of iterations the model should run
#runs = 5000
nagents = 100 #100
nglob = 5
s = 1.5 #0.5 #0.1
m = 1 #0.5
c = 0.8


jump_point = 1500
jump_size = 1000
#jump_param = 2
jump_sizes = np.arange(1000,10000, step=2000)#np.logspace(3, 5, num = 5) #np.arange(100, 2000, 100)
#jump_sizes = [1000,3000]
increases = rng_glod.uniform(0.1, 1, nglob)
increases = increases * c / np.sum(increases)
#increases = np.array([0.05,0.15,0.175,0.225,0.2])
print(increases)
print(c)

taus = rng_glod.uniform(10,20, size= (nagents,nglob))

model_params = {"nr_agents": nagents, 
                "nr_global_params": nglob, 
                "spending_amount": s,
                "global_quantities": np.full(nglob, 1000), 
                "quantity_threashold": rng_glod.uniform(2000,3000, size= (nagents,nglob)), #np.full((nagents,nglob), 1000),
                "currency_threshold": taus, #np.full((nagents,nglob), 100),
                "token_accounts": taus #np.full((nagents,nglob), 100) 
                }

model_params['mining_amounts'] = np.full(nglob, m)
increases = rng_glod.uniform(0.1, 1, nglob)
increases = increases * c / np.sum(increases)
model_params['quantities_increase'] = increases


def run_model(jump_size, rng, jump_param):

    # model_params = {"nr_agents": nagents, 
    #                 "initial_tokens": np.zeros(nglob), 
    #                 "nr_global_params": nglob, 
    #                 "spending_amount": s,
    #                 "global_quantities": np.full(nglob, 2500), 
    #                 "quantity_threashold": rng.uniform(2000,3000, size= (nagents,nglob)), #np.full((nagents,nglob), 1000),
    #                 "currency_threshold": taus, #np.full((nagents,nglob), 100),
    #                 "token_accounts": taus #np.full((nagents,nglob), 100) 
    #                 }

    # model_params['mining_amounts'] = np.full(nglob, m)
    # model_params['quantities_increase'] = increases
    
    #increases = np.array([0.05, 0.1, 0.15, 0.2, 0.3])

    # taus = rng.uniform(10,20, size= (nagents,nglob))

    # model_params = {"nr_agents": nagents, 
    #                 "initial_tokens": np.zeros(nglob), 
    #                 "nr_global_params": nglob, 
    #                 "spending_amount": s,
    #                 "global_quantities": np.full(nglob, 1000), 
    #                 "quantity_threashold": rng.uniform(2000,3000, size= (nagents,nglob)), #np.full((nagents,nglob), 1000),
    #                 "currency_threshold": taus, #np.full((nagents,nglob), 100),
    #                 "token_accounts": taus #np.full((nagents,nglob), 100) 
    #                 }
    

    # model_params['mining_amounts'] = np.full(nglob, m)
    # increases = rng.uniform(0.1, 1, nglob)
    # increases = increases * c / np.sum(increases)
    # model_params['quantities_increase'] = increases


    model = Model(model_params["nr_agents"], 
                  model_params["nr_global_params"], 
                  model_params["quantity_threashold"], 
                  model_params["currency_threshold"], 
                  model_params["token_accounts"], 
                  model_params["global_quantities"], 
                  model_params["quantities_increase"], 
                  model_params["mining_amounts"],
                  model_params["spending_amount"], rng)

    # Run the model
   
    nIts = int(jump_point + jump_size)
    #jump_param = rng.integers(0,nglob-1)
    print(jump_param)
    data = model.run(numIterations=nIts, printIt=False, critical_jumps=[jump_point], mean_jump_size=jump_size, jump_params = [jump_param])
    
    df = pd.DataFrame(data)
    df["jumpParam"] = jump_param
    df["c_jumpParam"] = increases[jump_param]
    return df

   
def call_model(seed): # (seed, c, m)
    rng = np.random.default_rng(seed)
    data = []
    jump_param = seed // 10
    for js in jump_sizes:
            print(seed, js)
            df = run_model(js, rng, jump_param)
            df["perturbation size"] = js
            df["rep"] = seed
            data.append(df)
    
    return pd.concat(data)

def main():

     # creating a pool
    p = multiprocessing.Pool() 
    # map list to target function 
    result = p.map(call_model, np.arange(numReps)) 
    pd.concat(result).to_csv("data/recovery_time_linear_bigS_50reps_100a_randJumpParam_s=1.5_new.csv")


    print(increases)

    print("finished")

    
if __name__ == "__main__":
    main()