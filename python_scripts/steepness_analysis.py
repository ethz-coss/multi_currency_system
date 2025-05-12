from Model import Model
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import pandas as pd
import sys
import csv
from os import path
 #TODO: parallelize


nreps = 5

"""
    Same script as "steepness_analysis_parallel" but serial version.
"""

rng = np.random.default_rng(seed=10)


def run_model(model_params, runs):
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
    #pd.DataFrame(model.run(runs))

    return model.run(runs)
    

def main():

    """ Defines the parameters of the model run, runs the model and plots results. """

    # Number of iterations the model should run
    runs = 1000

    init_global_params = 1000
    global_threashold = 1000
    token_threashold = 100
    token_accounts = 100

    #agentType = AgentType.DEFAULT

    #const. total increase of global quantities
    nagents = 100
    nglob = 4
    s = 0.6 # irrelevant

    c_tot_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    m_vals = c_tot_vals




    # pathtofile = Path.cwd()/'data'/'numagents'/filename
    # # writing to csv file  
    # with open(pathtofile, 'w') as csvfile:  
    #     # creating a csv writer object  
    #     csvwriter = csv.writer(csvfile)  

    #     # writing the fields  
    #     fields = ['number of agents', 'random seed', 'average utility of best design', 'average utility of voted design','smax', 'smin', 'number of created designs', 'utility param', 'number of same designs', 'diversity param', 'median']

    #     csvwriter.writerow(fields)  

    data = []

    seed = 0

    for c_tot in c_tot_vals:
        for m in m_vals:
            for r in range(nreps):
                print('c, m, r =', c_tot, m, r)
                #nagents = np.random.randint(2,50)
                #nglob = np.random.randint(2,10)
                increases = np.random.uniform(0.1,1,nglob)
                increases = increases * c_tot / np.sum(increases)

                model_params = {"nr_agents": nagents, "initial_tokens": np.zeros(nglob), "nr_global_params": nglob, 
                                     "mining_amounts": np.full(nglob, m), 
                                    "spending_amount": s,
                                  "global_quantities": np.full(nglob, init_global_params), 
                                "quantities_increase": increases, #np.random.uniform(0.05, 0.35, nglob), #np.full(nglob, 0.2), #  g_i += c_i after each step
                                "quantity_threashold": np.full((nagents,nglob), global_threashold),
                                "currency_threshold": np.full((nagents,nglob), token_threashold),
                                    "token_accounts": np.full((nagents,nglob), token_accounts), 
                                              "seed": seed}

                seed += 1

                model_data = pd.DataFrame(run_model(model_params, runs))

                model_data['c_tot'] = c_tot
                model_data['m'] = m
                model_data['r'] = r
                model_data['s'] = s
                model_data['G'] = nglob
                model_data['step'] = np.arange(0,runs)
                model_data['c1, ..., cG'] = str(increases)
                #model_data['agent threasholds'] = str(increases)
                data.append((model_data))


            #model_data.to_csv("model_data_c=" + str(c_tot) + "_m=" + str(m).csv")

        pd.concat(data).to_csv("steepness_vs_m_c_s=" + str(s) + ".csv")
        #print(data)
                
            # # writing the data rows  
            # for row in result:
            #     csvwriter.writerows(row) 

        
            #data.to_csv("c_m_data")
            #agent_data.to_csv("agent_data.csv")
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