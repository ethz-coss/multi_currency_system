from Model import Model
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import pandas as pd
import sys
import csv



"""
This script runs the model for a stable and unstable configuration.
The increase constant of one global parameter becomes bigger along a small time interval.

Output trajectories of the global parameters are stored in the file "increases_crisis.csv"

Output data is visualized in the file "crisis_figures.ipynb"

"""


def main():

  seed = 11
  rng = np.random.default_rng(seed)

  # Number of iterations the model should run
  runs = 4000
  nreps = 10 #10
  
  '''
  nglob = 4 #4
  c = 0.25 # 0.25

  nagents = 50

  model_params = {"nr_agents": nagents,  "nr_global_params": nglob, 
                                  "global_quantities": np.full(nglob, 0), 
                                "quantities_increase": np.full(nglob, c), #np.random.uniform(0.05, 0.35, nglob), #np.full(nglob, 0.2), #  g_i += c_i after each step
                  "quantity_threashold_lower_bounds": np.full(nglob, 10),
                  "quantity_threashold_upper_bounds": np.full(nglob, 500),
                                "tokens_lower_bound": np.full(nglob, 0), 
                                "tokens_upper_bound": np.full(nglob, 0),
                        "token_demands_lower_bounds": np.full(nglob, 10),
                        "token_demands_upper_bounds": np.full(nglob, 100)}
  '''
  # model_params = {"nr_agents": nagents,  "nr_global_params": nglob, 
  #                                "global_quantities": np.full(nglob, 800), 
  #                              "quantities_increase": np.full(nglob, c), #np.random.uniform(0.05, 0.35, nglob), #np.full(nglob, 0.2), #  g_i += c_i after each step
  #                 "quantity_threashold_lower_bounds": np.full(nglob, 10),
  #                 "quantity_threashold_upper_bounds": np.full(nglob, 400),
  #                               "tokens_lower_bound": np.full(nglob, 10), 
  #                               "tokens_upper_bound": np.full(nglob, 400),
  #                       "token_demands_lower_bounds": np.full(nglob, 10),
  #                       "token_demands_upper_bounds": np.full(nglob, 400)}




    #many global params
  '''  
  ## for eq. m = c * G
  nglob = 4

  model_params = {"nr_agents": 10,  "nr_global_params": nglob, 
                                  "global_quantities": np.full(nglob, 100), 
                                "quantities_increase": np.full(nglob, 0.1), #np.random.uniform(0.05, 0.35, nglob), #np.full(nglob, 0.2), #  g_i += c_i after each step
                    "quantity_threashold_lower_bounds": np.full(nglob, 10),
                    "quantity_threashold_upper_bounds": np.full(nglob, 10),
                                  "tokens_lower_bound": np.full(nglob, 100), 
                                  "tokens_upper_bound": np.full(nglob, 100),
                          "token_demands_lower_bounds": np.full(nglob, 80),
                          "token_demands_upper_bounds": np.full(nglob, 80)}
  '''   
  # Initialize model parameters
  # model_params = {                        "nr_agents": 50, 
  #                                  "nr_global_params": 4, 
  #                                 "global_quantities": [400,1000,50,50],  #initial amount of the global quantities
  #                  "quantity_threashold_lower_bounds": [0,0,0,0],         # threasholds for each agent uniformly random within bounds
  #                  "quantity_threashold_upper_bounds": [500,500,500,500],
  #                                "tokens_lower_bound": [0,0,0,0],         #token accounts uniformly random within bounds
  #                                "tokens_upper_bound": [400,1000,50,50],
  #                        "token_demands_lower_bounds": [0,0,0,0],         # token demands for each agent uniformly random within bounds
  #                        "token_demands_upper_bounds": [500,100,100,100]}

  # # #mining only: no needs for tokens
  # model_params = {"nr_agents": 10, "initial_tokens": [0, 0, 0, 0], "nr_global_params": 4, 
  #                                 "global_quantities": [700,600,600,600], 
  #                  "quantity_threashold_lower_bounds": [600,500,500,500],
  #                  "quantity_threashold_upper_bounds": [600,500,500,500],
  #                                "tokens_lower_bound": [0,0,0,0],
  #                                "tokens_upper_bound": [0,0,0,0],
  #                        "token_demands_lower_bounds": [0,0,0,0],
  #                        "token_demands_upper_bounds": [0,0,0,0]}
  
  # # #mining only: no needs for tokens

  
  #const. total increase of global quantities

  # #stable case
  nagents = 100
  nglob = 5
  m = 1.0 #0.6
  s = 0.5 #0.2
  c = 0.8 #0.75 #0.4
  # jump_point = 1000
  # jump_size = 1000

  #cstart = 1500
  #cend = 2500

  cstart = 1000
  cend = 2000

  incr_factors = [2.0, 3.0]
  incr_idx = 0

  total_data = []

  for r in range(nreps):
    for i in incr_factors:
      increases = np.full((runs, nglob), c/nglob)
      rng = np.random.default_rng(r)
      for t in range(cstart, cend):
        #increases[t, 0] *= 2.0
        increases[t, incr_idx] *= i

        

        

      print(np.max(increases))

      # increases = rng.uniform(0.1, 1, nglob)
      # increases = increases * c / np.sum(increases)
      # print(increases)

      model_params = {"nr_agents": nagents, "initial_tokens": np.zeros(nglob), "nr_global_params": nglob, 
                                        "mining_amounts": np.full(nglob, m), 
                                        "spending_amount": s,
                                      "global_quantities": np.full(nglob, 700), 
                                    "quantities_increase": np.full(nglob, c/nglob), #np.random.uniform(0.05, 0.35, nglob), #np.full(nglob, 0.2), #  g_i += c_i after each step
                                    "quantity_threashold": rng.uniform(500,1500, size= (nagents,nglob)), #np.full((nagents,nglob), 1000),
                                    "currency_threshold": rng.uniform(50,100, size= (nagents,nglob)), #np.full((nagents,nglob), 100),
                                        "token_accounts": rng.uniform(50,100, size= (nagents,nglob)) #np.full((nagents,nglob), 100) 
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
    
      data = model.run(numIterations=runs, printIt=True, dynamic_ci = True, incr_consts = increases)
      
      df = pd.DataFrame(data)

      df['c_vals'] = increases.tolist()
      df['increase factor'] = i
      df['rep'] = r
      df['step'] = np.arange(0, runs)

      total_data.append(df)

  pd.concat(total_data).to_csv("increases_crisis.csv")
  #model_data.to_csv("model_data.csv")


  # print("number of trades: ", model.numTrades)
  # #print(model.numTradeEvol)
  # print("G, A, c, m, s ")
  # print(nglob, nagents, c, m, s)
  # print("avg. tokens spent per agent and step (s): ", model.avg_spending_amounts / nagents / runs )
  # print("avg. tokens mined per agent and step (m): ", model.avg_mining_amounts / nagents / runs)
  # print("increase of glob params per agent and step (c): ", model.increases / model.numagents / runs)
  # print("avg. traded amounts per step: ", model.traded_amounts / runs)


  print("finished")

  
if __name__ == "__main__":
  main()