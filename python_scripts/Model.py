import numpy as np
from scipy.special import softmax
from enum import Enum

# Actions:
SLEEP = -1 # do nothing 
TRADE = 0
MINE = 1

class Model():
    """
    Implementation of a multi-currency system that is driven by individual needs of agents.

    Parameters:
        nagents: number of agents
        nglob: number of global parameters
        world_thresholds: array of size (nagents,nglob) with the thresholds gamma_a,i for the needs to reduce global parameters
        currency_thresholds: array of size (nagents,nglob) with the thresholds tau_a,i for the needs to have currencies
        initial_token_accounts: array of size (nagents,nglob) with the initial number of tokens that every agent has for each currency 
        initial_global_parameters: array of size (nglob) with the initial levels of all global parameters
        increase_constants: array of size (nglob) with the increase constants c_i of all global parameters
        mining_amounts: array of size (nglob) with the mining amounts m_i of every global parameter
        spending_amount: the amount that each agent spends per iteration of a random currency
        random_generator: the random generator from which random numbers are drawn
    """

    def __init__(self, nagents, nglob, world_thresholds, currency_thresholds, initial_token_accounts, initial_global_parameters, increase_constants, mining_amounts, spending_amount, random_generator) -> None:
        """ Initialization of the global parameters, thresholds and the agents
        """
        self.rng = random_generator 

        self.nagents = nagents
        self.nglob = nglob

        ### global parameters
        self.global_parameters = np.array(initial_global_parameters, dtype=float) # np.zeros((nglob))

        self.increase_constants = increase_constants
        self.mining_amounts = mining_amounts
        self.spending_amount = spending_amount

        ### parameters for all agents
        self.world_thresholds = np.array(world_thresholds, dtype=float) #np.zeros((nagents, nglob))
        self.currency_thresholds = np.array(currency_thresholds, dtype=float) # np.zeros((nagents, nglob))
        self.token_accounts = np.array(initial_token_accounts, dtype=float) # np.zeros((nagents, nglob))

        self.world_needs = np.zeros((nagents, nglob))
        self.currency_needs = np.zeros((nagents, nglob))

        self.actions = np.ndarray(self.nagents, dtype = int) # -1: no action, 0: trade, 1: mine
        self.active_currencies = np.ndarray(self.nagents, dtype = int)

        ### values of the currencies, used to compute the exchange rate
        self.values = np.zeros(self.nglob)
        self.offset = 0.01 #offset to avoid divinding by zero when computing exchange rates
        self.traded_amounts = np.zeros(self.nglob)


    def run(self, numIterations, printIt=False, critical_jumps = [], mean_jump_size = 100, jump_params = [], dynamic_ci = False, incr_consts = [], returnNeeds = False):
        """
            Input:
                numIterations: (int), number of iterations of one cycle of the model
                critical_jumps: (list), times where sudden critical events happen
                mean_jump_size: (int), mean size that a global parameter increases at a sudden critical event
                jump_params: (list), indices of the global parameters that increase at a sudden critical event
                dynamic_ci: (bool), if true, the increase constants c_i are set to the values from the list incr_consts.
                incr_consts: (list), only needed if dynamic_ci == true. Then, shape = (numIterations, nglob) and contains the increase constants of all global parameters for every timestep. 
                returnNeeds: (bool), if true, the mean needs on all agents are returned for each timestep and global parameter

            Returns:
                dict with:
                    'Agents': (int), number of agents
                    'Global quantities': (ndarray), shape = (numIterations, nglob) 
                    'token account': (ndarray), shape = (numIterations, num agents, nglob) 
                    'num trades': (ndarray), shape = (numIterations, total number of mining actions for each timestep 
                    'num mining': (ndarray), shape = (numIterations), total number of trading actions for each timestep 
                    'mean global needs': (ndarray), shape = (numIterations, nglob) 
                    'mean token needs': (ndarray), shape = (numIterations, nglob) 
            
        """

        
        ## create matrices to store data:
        globalParamEvol = np.ndarray((numIterations, self.nglob), dtype=float)
        MeanTokenAccountEvol = np.ndarray((numIterations, self.nglob), dtype=float)
        nMinesEvol = np.ndarray(numIterations, dtype=int) #num sleep, n trade, n mine
        nTradesEvol = np.ndarray(numIterations, dtype=int)

        if returnNeeds:
            meanGlobalNeeds = np.ndarray((numIterations, self.nglob), dtype=float)
            meanCurrencyNeeds = np.ndarray((numIterations, self.nglob), dtype=float)


        for t in range(numIterations):
            if printIt and t % 100 == 0:
                print('iteration', t, '/', numIterations)
            #collect data:
            globalParamEvol[t] = self.global_parameters
            MeanTokenAccountEvol[t] = np.mean(self.token_accounts, axis=0)
            if returnNeeds:
                meanGlobalNeeds[t] = np.mean(self.world_needs, axis=0)
                meanCurrencyNeeds[t] = np.mean(self.currency_needs, axis=0)
            
            ### simulate one model cycle
            if t in critical_jumps:
                numMines, numTrades = self.step(self.increase_constants, mean_jump_size, jump_params)
            else:
                if dynamic_ci:
                    numMines, numTrades = self.step(incr_consts[t], 0.0)
                else:
                    numMines, numTrades = self.step(self.increase_constants, 0.0)

            nMinesEvol[t] = numMines
            nTradesEvol[t] = numTrades
        
        data = {'Agents': self.nagents, 
                'Global quantities': globalParamEvol.tolist(), 
                'token account': MeanTokenAccountEvol.tolist(), 
                'num trades': nTradesEvol.tolist(),
                'num mining': nMinesEvol.tolist()}
        
        if returnNeeds:
            data['mean global needs'] = meanGlobalNeeds.tolist()
            data['mean currency needs'] = meanCurrencyNeeds.tolist()

        return data
    

    def run_withBothPerturbations(self, numIterations, critical_jump_times, mean_jump_size, c_variance):
        """
            Simulation of a sytem where sudden critical events happen at random timesteps 
            and additionally the increase constants fluctuate randomly by Brownian motion at every 10-th timestep.

            Input:
                numIterations: (int), number of iterations
                critical_jump_times: (array), timesteps where sudden critical events happen
                mean_jump_size: (double), mean size by which one random global parameter increases at a sudden critical event.
                c_variance: variance by which the increase constants fluctuate. c_i(10t+1) = c_i(10t) + delta, where delta is sampled from the normal distribution N(0, c_variance).


            Returns:
                dict with:
                    'Agents': (int), number of agents
                    'Global quantities': (ndarray), shape = (numIterations, nglob) 
                    'token account': (ndarray), shape = (numIterations, num agents, nglob) 
                    'num trades': (ndarray), shape = (numIterations, total number of mining actions for each timestep 
                    'num mining': (ndarray), shape = (numIterations), total number of trading actions for each timestep 
                
        """

        ## create matrices to store data:
        globalParamEvol = np.ndarray((numIterations, self.nglob), dtype=float)
        MeanTokenAccountEvol = np.ndarray((numIterations, self.nglob), dtype=float)
        nMinesEvol = np.ndarray(numIterations, dtype=int) #num sleep, n trade, n mine
        nTradesEvol = np.ndarray(numIterations, dtype=int)

        total_increase_constants = np.ndarray(numIterations, dtype=float)

        for t in range(numIterations):
            # if  t % 100 == 0:
            #     print('iteration', t, '/', numIterations)
            #critical events:

            #increase constants change randomly like brownian motion
            #c_perturb = self.rng.normal(0.0, c_variance, size = self.nglob)
            if (t % 10 == 0): 
                self.increase_constants += self.rng.normal(0.0, c_variance, size = self.nglob)
                for gid in range(self.nglob):
                    if self.increase_constants[gid] < 0.0: self.increase_constants[gid] = 0

            total_increase_constants[t] = np.sum(self.increase_constants)

            numMines, numTrades = self.step(self.increase_constants, 0.0)
                
        
            if t in critical_jump_times:
                id = self.rng.integers(0, self.nglob)
                self.global_parameters[id] = self.global_parameters[id] +  self.rng.uniform(mean_jump_size/2, mean_jump_size + mean_jump_size/2)

            #collect data:
            globalParamEvol[t] = self.global_parameters
            MeanTokenAccountEvol[t] = np.mean(self.token_accounts, axis=0)
            nMinesEvol[t] = numMines
            nTradesEvol[t] = numTrades
        
        data = {'Agents': self.nagents, 
                'Global quantities': globalParamEvol.tolist(), 
                'token account': MeanTokenAccountEvol.tolist(), 
                'num trades': nTradesEvol.tolist(),
                'num mining': nMinesEvol.tolist(),
                'total increase constant': total_increase_constants.tolist()}


        return data

    def step(self, increase_chonstants, external_events = 0.0, jump_params = []):
        """
            executes one iteration step of the model.

            Input:
                increase_chonstants: (array), shape = (nglob), increase constants of all global parameters
                jump_params: (list), indices of all global parameters for which a sudden critical event happens
                external_events: (double), amount by which all global parameters from jump_params increase

            Returns:
                numMines: (double) total number of mining actions
                numTrades: (double) total number of trading actions

        
        """
        
        self.increase_global_parameters(increase_chonstants, external_events, jump_params)
        
        self.update_needs()
        self.update_values()

        for agent in range(self.nagents):
            self.agent_chooses_currency_and_action(agent)

        numTrades = self.perform_trading()

        numMines = self.perform_mining(self.mining_amounts) 
      
        self.spend_tokens(self.spending_amount) 

        return numMines, numTrades


        
    
    def increase_global_parameters(self, increase_constants, external_events=0.0, jump_params=[]):
        # Increase proportional to the number of agents
        self.global_parameters = self.global_parameters + float(self.nagents) * increase_constants
        
        # Additionally increase a global parameters for which a critical event happens:
        for id in jump_params:
            self.global_parameters[id] += external_events
        


    def update_needs(self):
        # compute the needs of the agents
        self.world_needs = np.maximum(0.0, self.global_parameters - self.world_thresholds)
        self.currency_needs = np.maximum(0.0, self.currency_thresholds - self.token_accounts)

    def update_values(self):
        # compute the value of each currency
        self.values = np.mean(self.currency_needs, axis = 0)

    def exchange_rate(self, c1, c2): 
        """
        Input: indices of two currencies c1, c2
        Returns: exchange rate r between c1 and c2: v(c1) = r * c(c2)
        """

        return (self.offset + self.values[c1]) / (self.offset + self.values[c2])

    def agent_chooses_currency_and_action(self, agent):
        my_world_needs = self.world_needs[agent]
        my_currency_needs = self.currency_needs[agent]
        highest_needs = np.maximum(my_world_needs, my_currency_needs)
   
        # choose currency
        #print('probs',highest_needs)
        #print(self.nglob)
        active_currency = self.rng.choice(self.nglob, p = softmax(highest_needs))
        #print('active currency: ', active_currency)

        # chooce action:
        my_action = TRADE

        if highest_needs[active_currency] <= 0.0:
            my_action = SLEEP
        
        elif my_world_needs[active_currency] >= my_currency_needs[active_currency]:
            my_action = MINE

        else:
            p = self.rng.uniform()
            if p < 0.5:
                my_action = MINE
            

        self.active_currencies[agent] = active_currency
        self.actions[agent] = my_action
        
    
    def check_if_trading_possible(self, a, b):
        
        if self.actions[a] != TRADE or self.actions[b] != TRADE:
            return False

        ca = self.active_currencies[a]
        cb = self.active_currencies[b]

        if ca == cb: return False

        tol = 1e-3
        if self.token_accounts[a, cb] < tol or self.token_accounts[b, ca] < tol: return False # the agents have of the partner the other wants
        
        return (self.currency_needs[a,ca] > self.currency_needs[a,cb] and
                self.currency_needs[b,cb] > self.currency_needs[b,ca])   
            

    def perform_trading(self):
        # 1. get all agents that want to trade
        traders = np.where(self.actions == TRADE)[0]
        numTrades = 0
        #shuffle traders s.t. random trading order
        self.rng.shuffle(traders)

        # perform trading: 
        for i in range(len(traders) - 1): # -1 because last trader can't trade since no more partners available
            a = traders[i]
            possible_partners = []
             
            for b in traders[i+1:]:
                if self.check_if_trading_possible(a,b):
                    possible_partners.append(b)


            # If no coutnerpart was found, the agent resorts to mining. Otherwise, the trade partner is chosen randomly from the counterpart list
            if (len(possible_partners) == 0):
                self.actions[a] = MINE
            
            else:
                numTrades += 1
                b = self.rng.choice(possible_partners)
                self.trade_with_exchange_rates(a, b)
                self.actions[b] = SLEEP
                self.actions[a] = SLEEP

        return numTrades
        
    def trade_with_exchange_rates(self, a, b):
        ### Perform trading between two possible trade partners a and b

        ca = self.active_currencies[a]
        cb = self.active_currencies[b]
        #otherToken = counterpart.active_token
        #print('trade is possible: ', self.check_if_trading_possible(a, b))
        #1. get trade amounts that giver and receiver are willing to exchange
        proposal_a = self.get_trade_amount(a, ca, cb) # (amount receiving of my token, amount willing to give of counterpartToken)
        proposal_b = self.get_trade_amount(b, cb, ca) # (amount receiving of counterpart token, amount willing to give of myToken)

        # print("trade proposals: ")
        # print(np.array(proposal_a), np.array(proposal_b))
        
        #2. get the trade amounts that both agents are willing to trade
        delta_ca = min(proposal_a[0], proposal_b[1])
        delta_cb = self.exchange_rate(cb, ca) * delta_ca

        # if PRINT_ALL:
        #     print("Trading between tokens", myToken, "and", otherToken, "with exchange rate", self.model.exchange_rate(otherToken, myToken))
        # if DEBUG:
        #     print("trade amounts (my token, counterpart token): ", myTokenAmount, counterpartTokenAmount)

        #3. transaction of the tokens
        self.token_accounts[a, ca] += delta_ca
        self.token_accounts[b, ca] -= delta_ca

        self.token_accounts[a, cb] -= delta_cb
        self.token_accounts[b, cb] += delta_cb

        #collect data:
        self.traded_amounts[ca] += delta_ca
        self.traded_amounts[cb] += delta_cb

    def get_trade_amount(self, agent, receiverToken, giverToken):
        ### compute token amount that the agent wants to give from the "giverToken" currency in order to get from the "receiverToken" currency. Returns the tuple (amount of receiverToken, amount of giverToken)

        exchange_rate = self.exchange_rate(receiverToken, giverToken)
        #print("exchange rate: " , exchange_rate)

        #choose amount s.t. after trade the needs for the receiver and giver Token are equal
        n_giver    = self.currency_thresholds[agent,giverToken]    - self.token_accounts[agent, giverToken] #need_threasholds[token][1] - giver.token_accounts[token] #giver.token_accounts[token] - giver.needs_functions[token][1] #should be negative (guaranteed? TODO: only if agent gets active when need above some percentage (e.g. > 20%) )
        n_receiver = self.currency_thresholds[agent,receiverToken] - self.token_accounts[agent, receiverToken] #because can be smaller than 0
        
        #test that n_giver < n_receiver:
        if n_giver > n_receiver:
            print("ERROR in GET_TRADE_AMOUNT: need of giver token bigger than receiver", n_giver, n_receiver)

        # amounts that need to be transacted s.t. afterwards the agents need is balanced
        trade_amount_giverToken = min((n_receiver - n_giver)/(1+exchange_rate), self.token_accounts[agent, giverToken])

        if trade_amount_giverToken <= 0.0:
            print("NEGATIVE TRADE AMOUNT", trade_amount_giverToken)
            trade_amount_giverToken = 0.0

        return (exchange_rate * trade_amount_giverToken, trade_amount_giverToken) #(amount receiving, amount giving)
                
    
    def mine(self, agent, gidx, mining_amounts):
        ## agent mines the global parameter with index gidx. 
        ## mining_amounts is a list that contains the mining_amounts of all global parameters. 
        amount = min(mining_amounts[gidx], self.global_parameters[gidx])
        self.global_parameters[gidx] -= amount
        self.token_accounts[agent, gidx] += amount


    def perform_mining(self, mining_amounts): 
        ## all agents that choose to mine execute the mining.
        numMines = 0
        for a in range(self.nagents):
            if self.actions[a] == MINE:
                numMines += 1
                self.mine(a, self.active_currencies[a], mining_amounts)
        return numMines

    def spend_tokens(self, spending_amount): 
        ## all agents spend "spending_amount" tokens of a random currency
        #         
        for a in range(self.nagents):
            ia = self.rng.choice(range(self.nglob), size=1) #indices[a]
            self.token_accounts[a, ia] = max(0.0, self.token_accounts[a, ia] - spending_amount) # np.max(0.0, self.token_accounts[a, ia] - spending_amount)


 

# if __name__ == '__main__':

#     nagents = 10
#     nglob = 2
#     initial_global_parameters = np.full(nglob, 50.0)
#     increase_constants = np.full(nglob, 1.0)
#     mining_amounts = np.full(nglob, 1.0)
#     spending_amount = 0.0

#     world_thresholds = np.full((nagents,nglob), 100)
#     currency_thresholds = np.full((nagents, nglob), 0)
#     initial_token_accounts = np.full((nagents, nglob), 0)

#     model = Model(nagents, nglob, world_thresholds, currency_thresholds, initial_token_accounts, initial_global_parameters, increase_constants, mining_amounts, spending_amount)
#     output = model.run(numIterations=50)
#     #print(output)
#    # print(output['Global quantities'])
#     #print(output['num mining'])