from Model import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import csv



"""
Runs 5 realizations of a stable and unstable system with G=5 global parameters and m=1.

Trajectories of the global parameters are stored in the files 
    "case_G=5_c=0.8.csv" (stable system)
    "case_G=5_c=1.3.csv" (unstable system)

Results of this script are visualized in the file "example_scenarions.ipynb"
    
"""

# class RunType(Enum):
#     DEFAULT = 1
#     NO_MINING = 2
    

def main():


    # Number of iterations the model should run
    runs = 4000
    nreps = 1

    init_global_params = 1000
    global_threashold = 2000
    token_threashold = 10
    token_accounts = 0.0

    # init_global_params = 3000
    # global_threashold = 2000
    # token_threashold = 0
    # token_accounts = 0.0

    
    #const. total increase of global quantities
    nagents = 100

    #cases: [nagents, nglob, m, s, c]
  
    scenarios = [[100, 5, 1.0, 0.6,  np.array([0.05, 0.1, 0.15, 0.2, 0.8])],    #stable case with params going to 0 and to gamma
                 [100, 5, 1.0, 0.6,  np.array([0.05, 0.1, 0.15, 0.2, 0.3])],  #case 2: s < c < m
                 [100, 2, 1.0, 0.6,  np.array([0.05, 0.95])]
                 #[100, 5, 1.0, 1.0,  np.array([0.05, 0.1, 0.15, 0.3, 0.6])]   #case 3: c < m < s    
                 #[100, 5, 1.0, 1.0, 1.0]

                ]
    


    # #case 1: s < m < c
    # nglob = 1
    # m = 1.0
    # s = 0.5
    # c = 1.5 # threashold for stability is mining_amount

    # #case 2: s < c < m
    # nglob = 1
    # m = 1.0
    # s = 0.5
    # c=  0.9 # threashold for stability is mining_amount

    # #case 3: c < m < s
    # nglob = 1
    # m = 1.0
    # s = 1.1
    # c=  0.9 # threashold for stability is mining_amount


    #case 5: case 1 + 2 mit N = 1000
    #case 4: case 1 + 2 mit G = 4


    seed = 0
    rng = np.random.default_rng(seed)

    for case in scenarios: 
        data = []
        nagents = case[0]
        nglob = case[1]
        m = case[2]
        s = case[3]
        increases = case[4]

        for r in range(nreps):
            seed += 1
            print('case: ', case, ', rep: ', r)

            # random increase constants s.t. sum equal to c:

            #increases = rng.uniform(0.1, 1, nglob)
            #increases = increases * c / np.sum(increases)

            gammas = np.full((nagents, nglob), global_threashold)
            # gamma_glob = rng.uniform(global_threashold, global_threashold + 1000, nglob)
            # gammas = np.zeros((nagents, nglob))
            # for i in range(nglob):
            #     gammas[:,i] = gamma_glob[i]

            taus = np.full((nagents,nglob), token_threashold)
    

            #increases = increases * c_sum / np.sum(increases)
            #increases = np.full(nglob, c) # every param incr the same

            model_params = {"nr_agents": nagents, "initial_tokens": np.zeros(nglob), "nr_global_params": nglob, 
                                     "mining_amounts": np.full(nglob, m), 
                                    "spending_amount": s,
                                  "global_quantities": np.full(nglob, init_global_params), 
                                "quantities_increase": increases, #np.random.uniform(0.05, 0.35, nglob), #np.full(nglob, 0.2), #  g_i += c_i after each step
                                "quantity_threashold": gammas, #np.full((nagents,nglob), global_threashold),
                                "currency_threshold": taus,
                                    "token_accounts": np.full((nagents,nglob), token_accounts)}



            
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
            
            model_data = pd.DataFrame(model.run(runs, returnNeeds = True))

            model_data['r'] = r
            model_data['c'] = np.sum(increases)
            model_data['m'] = m
            model_data['G'] = nglob
            model_data['s'] = s
            model_data['step'] = np.arange(0,runs)
            model_data['c1, ..., cG'] = str(increases)
            model_data['N,G,m,s,c'] = str(case)
            model_data['global thresholds'] = global_threashold
            model_data['token thresholds'] = token_threashold


            #compute needs:


            #model_data['agent threasholds'] = str(increases)
            data.append((model_data))

        pd.concat(data).to_csv("case_G=5_c=" + str(np.sum(increases)) + ".csv")

    #model_data.to_csv("model_data_c=" + str(c_tot) + "_m=" + str(m).csv")

    #pd.concat(data).to_csv("example_scenarios_G=5.csv")
      


    print("finished")

    
if __name__ == "__main__":
    main()