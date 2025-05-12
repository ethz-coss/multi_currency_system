from Model import Model
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import pandas as pd
import sys
import csv



"""
    Run this file to run one realization of the whole model. 
    The output is stored in the file "model_data.csv"

"""


def main():

    seed = 1
    rng = np.random.default_rng(seed)
    """ Defines the parameters of the model run, runs the model and plots results. """

    # Number of iterations the model should run
    runs = 2000

   
    # #lin increasing case
    nagents = 100
    nglob = 3
    m = 1.0 
    s = 0.5

    #increase constants
    increases = np.array([0.5, 0.2, 0.1])
  

    #times and size of sudden critical events.
    jump_points = [500, 1500]
    jump_size = 5000

  
    # sampling random thresholds gamma for all global parameters that are the same among agents.
    gamma_glob = np.random.uniform(1000, 2000, nglob)
    gammas = np.zeros((nagents, nglob))
    for i in range(nglob):
        gammas[:,i] = gamma_glob[i]
    


    model_params = {"nr_agents": nagents, "initial_tokens": np.zeros(nglob), "nr_global_params": nglob, 
                "mining_amounts": np.full(nglob, m), 
              "spending_amount": s,
            "global_quantities": np.full(nglob, 1000), # np.random.uniform(100, 5000, nglob), 
          "quantities_increase": increases, 
          "quantity_threashold": gammas, #rng.uniform(2000,1500, size =(nagents,nglob)), #np.full((nagents,nglob), 3000),
            "currency_threshold": np.full((nagents,nglob), 0), #rng.uniform(50,100, size= (nagents,nglob)),
                "token_accounts": np.zeros((nagents, nglob)) #rng.uniform(50,100, size= (nagents,nglob)) #np.full((nagents,nglob), 100) 
                }


    
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
   
    data = model.run(numIterations = runs, printIt = True, critical_jumps = jump_points, mean_jump_size = jump_size)
    
    df = pd.DataFrame(data)

    df.to_csv("model_data.csv")


    print("finished")

    
if __name__ == "__main__":
    main()