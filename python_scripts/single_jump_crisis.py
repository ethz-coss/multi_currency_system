from Model import Model
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import pandas as pd
import sys
import csv



"""

This script runs a stable and unstable model where a sudden jump of one global parameters occurs

Output is stored in the files "stable_single_crisis.csv", "unstable_single_crisis.csv" and visualized in the file "crisis_figures.ipynb"

"""

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


def main():

    seed = 11
    rng = np.random.default_rng(seed)
    """ Defines the parameters of the model run, runs the model and plots results. """

    # Number of iterations the model should run
    runs = 2000
    nagents = 100
    nglob = 5
    s= 0.1

    model_params = {"nr_agents": nagents, "initial_tokens": np.zeros(nglob), "nr_global_params": nglob, 
                                      "spending_amount": s,
                                    "global_quantities": np.full(nglob, 2500), 
                                  "quantity_threashold": rng.uniform(2000,3000, size= (nagents,nglob)), #np.full((nagents,nglob), 1000),
                                   "currency_threshold": rng.uniform(50,100, size= (nagents,nglob)), #np.full((nagents,nglob), 100),
                                       "token_accounts": rng.uniform(50,100, size= (nagents,nglob)) #np.full((nagents,nglob), 100) 
                                       }

    #unstable linear increasing case at equilibrium (m > ctot)
    m = 0.5
    c = 0.7
  
    jump_point = 500
    jump_size = 1000
    jump_param = 0

    model_params['mining_amounts'] = np.full(nglob, m)
    increases = rng.uniform(0.1, 1, nglob)
    increases = increases * c / np.sum(increases)
    model_params['quantities_increase'] = increases

    data_unstable = run_model(model_params, jump_size, jump_point, jump_param, rng, runs)
    data_unstable.to_csv("unstable_single_crisis.csv")


    ####### crisis in stable system (m < ctot) ####
    m = 0.5
    c = 0.3 #0.4
  
    jump_point = 500
    jump_size = 1000
    jump_param = 0

    model_params['mining_amounts'] = np.full(nglob, m)
    increases = rng.uniform(0.1, 1, nglob)
    increases = increases * c / np.sum(increases)
    model_params['quantities_increase'] = increases

    data_unstable = run_model(model_params, jump_size, jump_point, jump_param, rng, runs)
    data_unstable.to_csv("stable_single_crisis.csv")
    
    # model = Model(model_params["nr_agents"], 
    #               model_params["nr_global_params"], 
    #               model_params["quantity_threashold"], 
    #               model_params["currency_threshold"], 
    #               model_params["token_accounts"], 
    #               model_params["global_quantities"], 
    #               model_params["quantities_increase"], 
    #               model_params["mining_amounts"],
    #               model_params["spending_amount"], rng)

    # # Run the model
   
    # data = model.run(numIterations=runs, printIt=True, critical_jumps=[jump_point], mean_jump_size=jump_size)
    
    # df = pd.DataFrame(data)




    # #m=c increasing case
    # nagents = 100
    # nglob = 5
    # m = 0.5
    # s = 0.2
    # c = 0.4 # 0.55
    # jump_point = 2000
    # jump_size = 2000


    # #case 1: s < m < c
    # nglob = 1
    # m = 1.0
    # s = 0.5
    # c=  1.5 # threashold for stability is mining_amount

    # #case 2: s < c < m
    # nglob = 1
    # m = 1.0
    # s = 0.5
    # c=  0.9 # threashold for stability is mining_amount

    #case 3: c < m < s
    # nglob = 4
    # m = 1.0
    # s = 0.8
    # c = 0.6 # threashold for stability is mining_amount



    #case 5: case 1 + 2 mit N = 1000
    #case 4: case 1 + 2 mit G = 4


    # mining_amounts = (0.1, 1, nglob)
    # mining_amounts = rng.normal(m, size=nglob) #mining_amounts * (nglob * m) / np.sum(mining_amounts)
    # print(np.mean(mining_amounts))

    #df.to_csv("model_data.csv")
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