from Model import Model
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import pandas as pd
import sys
import csv



"""
This script runs a model where sudden jumps of global parameters occur and 
the increase constants fluctuate based on a random walk.

Output is stored in the file "random_crisis.csv" and visualized in the file "crisis_figures.ipynb"
"""


def main():

  reps = [7] #[1,2,3,7]
  runs = 8000
  nagents = 100
  nglob = 5
  m = 1.0
  s = 0.5 #0.2
  c = 1.0
  c_variance = 0.01 # random walk of increase constants
  #c changes every 10-th. step

  minGamma = 500
  maxGamma = 1500
  g_init = 200

  numJumps = 10
  meanJumpSize = maxGamma * 2.0 #0.5 # g_i = g_i + uniform(meanJumpSize/2, meanJumpSize + meanJumpSize/2) 

  ## idea: incr factors random walk?

  incr_factors = [2.0, 3.0]
  incr_idx = 0

  total_data = []

  for r in reps:
      rng = np.random.default_rng(r)
      incrConsts = rng.uniform(0.1,1.0, size=nglob)
      incrConsts = incrConsts / np.sum(incrConsts) * c

      gamma_glob = rng.uniform(minGamma, maxGamma, nglob)
      gammas = np.zeros((nagents, nglob))
      for idx in range(nglob):
          gammas[:,idx] = gamma_glob[idx]
    # for i in incr_factors:
    #   incrConsts = np.full((runs, nglob), c/nglob)
    #   for t in range(cstart, cend):
    #     #increases[t, 0] *= 2.0
    #     increases[t, incr_idx] *= i

    #   print(np.max(increases))

      ### random critical jumps:

      model_params = {"nr_agents": nagents, "initial_tokens": np.zeros(nglob), "nr_global_params": nglob, 
                                        "mining_amounts": np.full(nglob, m), 
                                        "spending_amount": s,
                                      "global_quantities": np.full(nglob, g_init), 
                                    "quantities_increase": incrConsts, #np.random.uniform(0.05, 0.35, nglob), #np.full(nglob, 0.2), #  g_i += c_i after each step
                                    "quantity_threashold": gammas, #np.full((nagents,nglob), 1000),
                                    "currency_threshold": rng.uniform(50,100, size= (nagents,nglob)), #np.full((nagents,nglob), 100),
                                        "token_accounts": rng.uniform(50,100, size= (nagents,nglob)) #np.full((nagents,nglob), 100) 
                                        }

      for nJumps in [0, numJumps]:
        critical_jump_times = [] #rng.uniform(0, runs, size=numJumps)
        if nJumps != 0:
          critical_jump_times = rng.integers(0, runs, size=numJumps)
          #critical_jump_times = rng.uniform(0, runs, size=numJumps)
        # increases = rng.uniform(0.1, 1, nglob)
        # increases = increases * c / np.sum(increases)
        # print(increases)

        print(critical_jump_times)

        
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
        print('rep: ', r)
        data = model.run_withBothPerturbations(runs, critical_jump_times, meanJumpSize, c_variance)
        
        df = pd.DataFrame(data)
        df['rep'] = r
        df['step'] = np.arange(0, runs)
        df['num jumps'] = nJumps
    

        total_data.append(df)

  pd.concat(total_data).to_csv("random_crisis.csv")
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