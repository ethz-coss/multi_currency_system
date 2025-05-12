from Model import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import csv



"""
This script runs the model for different numbers of global parameters, 
to study, how the number of global parameters impacts the stability.

The output is stored in the file "example_scenarios_G_impact.csv"
Results of this script are visualized in the file "example_scenarions.ipynb"


"""

# class RunType(Enum):
#     DEFAULT = 1
#     NO_MINING = 2
    

def main():

    # Number of iterations the model should run
    runs = 300
    nreps = 30


    init_global_params = 1500
    global_threashold = 1000
    token_threashold = 10
    token_accounts = 10

    
    #const. total increase of global quantities
    nagents = 100

    #cases: [nagents, nglob, m, s, c]
    s = 0.0
    c = 0.25
    m = 1.0

    scenarios = [  #case 2: s < c < m
                 [100, 2, m, s, c], 
                 [100, 3, m, s, c], 
                 [100, 4, m, s, c], 
                 [100, 5, m, s, c],
                 [100, 6, m, s, c]
                 #[100, 5, 1.0, 0.5, 1.1],    #case 4: case 1 + 2 mit G = 4 
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

    data = []

    seed = 10
    rng = np.random.default_rng(seed)

    for case in scenarios: 
        nagents = case[0]
        nglob = case[1]
        m = case[2]
        s = case[3]
        c = case[4]

        for r in range(nreps):
            seed += 1
            print('case: ', case, ', rep: ', r)
            
            #increases = increases * c_sum / np.sum(increases)
            increases = np.full(nglob, c) # every param incr the same

            model_params = {"nr_agents": nagents, "initial_tokens": np.zeros(nglob), "nr_global_params": nglob, 
                                     "mining_amounts": np.full(nglob, m), 
                                    "spending_amount": s,
                                  "global_quantities": np.full(nglob, init_global_params), 
                                "quantities_increase": increases, #np.random.uniform(0.05, 0.35, nglob), #np.full(nglob, 0.2), #  g_i += c_i after each step
                                "quantity_threashold": np.full((nagents,nglob), global_threashold),
                                "currency_threshold": np.full((nagents,nglob), token_threashold),
                                    "token_accounts": np.full((nagents,nglob), token_accounts) }



            
            
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
            
            model_data = pd.DataFrame(model.run(runs))
          

            model_data['r'] = r
            model_data['c'] = c
            model_data['m'] = m
            model_data['G'] = nglob
            model_data['s'] = s
            model_data['step'] = np.arange(0,runs)
            model_data['c1, ..., cG'] = str(increases)
            model_data['N,G,m,s,c'] = str(case)
            model_data['global thresholds'] = global_threashold
            model_data['token thresholds'] = token_threashold
            #model_data['agent threasholds'] = str(increases)
            data.append((model_data))


    #model_data.to_csv("model_data_c=" + str(c_tot) + "_m=" + str(m).csv")

    pd.concat(data).to_csv("example_scenarios_G_impact.csv")
      


    print("finished")

    
if __name__ == "__main__":
    main()