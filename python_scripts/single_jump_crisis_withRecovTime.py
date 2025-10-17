from Model import Model
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import pandas as pd
import sys
import csv
import multiprocessing


"""

This script runs a stable and unstable model where a sudden jump of one global parameters occurs

Output is stored in the files "stable_single_crisis.csv", "unstable_single_crisis.csv" and visualized in the file "crisis_figures.ipynb"

"""

 # Number of iterations the model should run
runs = 2000
nagents = 100
nglob = 5
s = 0.5 #0.1
m = 1 #0.5
jump_sizes = np.arange(1000,10000, step=2000)#np.logspace(3, 5, num = 5) #np.arange(100, 2000, 100)

numReps = 10

jump_point = 1000
jump_size = 1000
jump_param = 1


def run_model(model_params, jump_size, jump_point, jump_param, rng, runs):
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
   
    data = model.run(numIterations=runs, printIt=True, critical_jumps=[jump_point], mean_jump_size=jump_size, jump_params = [jump_param])
    
    df = pd.DataFrame(data)
    return df


def call_model(seed): # (seed, c, m)
    rng = np.random.default_rng(seed)
    data = []

    for js in jump_sizes:
            print(seed, js)
            df = run_model(model_params,js,jump_point, jump_param, rng, runs)
            df["perturbation size"] = js
            df["rep"] = seed
            data.append(df)
    
    return pd.concat(data)

def main():

    seed = 11
    rng = np.random.default_rng(seed)
    """ Defines the parameters of the model run, runs the model and plots results. """

    # # Number of iterations the model should run
    # runs = 2000
    # nagents = 100
    # nglob = 5
    # s = 0.5 #0.1
    # m = 1 #0.5

    model_params = {"nr_agents": nagents, "initial_tokens": np.zeros(nglob), "nr_global_params": nglob, 
                                      "spending_amount": s,
                                    "global_quantities": np.full(nglob, 1000), 
                                  "quantity_threashold": rng.uniform(2000,3000, size= (nagents,nglob)), #np.full((nagents,nglob), 1000),
                                   "currency_threshold": rng.uniform(10,20, size= (nagents,nglob)), #np.full((nagents,nglob), 100),
                                       "token_accounts": rng.uniform(10,20, size= (nagents,nglob)) #np.full((nagents,nglob), 100) 
                                       }

    #unstable linear increasing case at equilibrium (m < ctot)
    #m = 0.5
    c = 1.2
  
    # jump_point = 1000
    # jump_size = 1000
    # jump_param = 1

    model_params['mining_amounts'] = np.full(nglob, m)
    increases = rng.uniform(0.1, 1, nglob)
    increases = increases * c / np.sum(increases)
    model_params['quantities_increase'] = increases

    data_unstable = run_model(model_params, jump_size, jump_point, jump_param, rng, runs)
    data_unstable.to_csv("data/unstable_single_crisis.csv")


    ####### crisis in stable system (m < ctot) ####
    #m = 0.5
    c = 0.8 #0.3 #0.4
  
    # jump_point = 1000
    # jump_size = 1000
    # jump_param = 1

    model_params['mining_amounts'] = np.full(nglob, m)
    increases = rng.uniform(0.1, 1, nglob)
    increases = increases * c / np.sum(increases)
    model_params['quantities_increase'] = increases

    #data_stable = run_model(model_params, jump_size, jump_point, jump_param, rng, runs)
    #data_stable.to_csv("data/stable_single_crisis.csv")
    

#### recovery time for stable model:

    p = multiprocessing.Pool() 
    # map list to target function 
    result = p.map(call_model, np.arange(numReps)) 
    pd.concat(result).to_csv("data/stable_single_crisis_manyJS_10reps.csv")

    print("finished")

    
if __name__ == "__main__":
    main()