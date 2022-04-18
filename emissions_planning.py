import argparse
import fair
import gym
import json
import math
import os
import random
import sys
import time
import torch

import matplotlib.pyplot as plt
import numpy as np

from fair.RCPs import rcp26, rcp45, rcp60, rcp85
from functools import partial
from scipy.stats import gamma
from stable_baselines3 import PPO, DDPG, A2C


# Labels for plots
VARS = [
    "Temperature anomaly (ºC)", 
    "CO2 Emissions (GtC)", 
    "CO2 Concentration (ppm)", 
    "Radiative forcing (W m^-2)",
    "Reward "
]
MULTIGAS_VARS = [
    "Temperature anomaly (ºC)", 
    "CO2 Emissions (GtC)", 
    "CO2 Concentration (ppm)", 
    "CO2 forcing (W m^-2)",
    "CH4 forcing (W m^-2)",
    "N2O forcing (W m^-2)",
    "All other well-mixed GHGs forcing (W m^-2)",
    "Tropospheric O3 forcing (W m^-2)",
    "Stratospheric O3 forcing (W m^-2)",
    "Stratospheric water vapour from CH4 oxidation forcing (W m^-2)",
    "Contrails forcing (W m^-2)",
    "Aerosols forcing (W m^-2)",
    "Black carbon on snow forcing (W m^-2)",
    "Land use change forcing (W m^-2)",
    "Volcanic forcing (W m^-2)",
    "Solar forcing (W m^-2)",
    "Reward "
]


#### Reward function options ####

def simple_reward(state, cur_temp, t, cur_emit, cur_conc):
    # positive reward for temp decrease
    # negative cliff if warming exceeds 2º
    if cur_temp > 2:
        return -100
    return (state[0] - cur_temp)

def temp_reward(state, cur_temp, t, cur_emit, cur_conc):
    # positive reward for temp under 1.5 goal
    if cur_temp > 2:
        return -100
    return 1.5 - cur_temp

def conc_reward(state, cur_temp, t, cur_emit, cur_conc):
    # positive reward for decreased concentration
    if cur_temp > 2:
        return -100
    return state[2] - cur_conc

def carbon_cost_reward(state, cur_temp, t, cur_emit, cur_conc):
    # impose a cost for each GtC emitted
    if cur_temp > 2:
        return -100
    return -cur_emit

def temp_emit_reward(state, cur_temp, t, cur_emit, cur_conc):
    # positive reward for keeping the temp under 1.5
    # negative reward for amount of emissions reduction
    # positive cliff for success at the end of the trial
    # w could indicate cost
    if cur_temp > 2:
        return -100
    if t==79 and temp <=1.5:
        return 100
    temp = 10*(state[0] - cur_temp)
    emit = state[1] - cur_emit
    if cur_emit < state[1]:
        return temp - emit
    return temp

def temp_emit_diff_reward(state, cur_temp, t, cur_emit, cur_conc):
    # positive reward for keeping the temp under 1.5
    # negative reward for amount of emissions reduction
    # (reduction compared to projected amount for that year)
    # positive cliff for success at the end of the trial
    # w could indicate cost of emissions
    if cur_temp > 2:
        return -100
    if t==79 and temp <=1.5:
        return 100
    curval = t*0.6 + 36
    temp = 10*(state[0] - cur_temp)
    emit = curval - cur_emit
    if cur_emit < curval:
        return temp - emit
    return temp


#### Environment for FaIR simulator ####
# built to run with the OpenAI Gym

class Simulator(gym.Env):

    def __init__(
        self,
        verbose=1, 
        action_space=36, 
        reward_mode="simple", 
        forcing=False,
        multigas=False,
        scenario='rcp60'
    ):
        # action space for the environment,
        # the amount to increase or decrease emissions by
        self.action_space = gym.spaces.Box(
            np.array([-action_space]).astype(np.float32),
            np.array([+action_space]).astype(np.float32),
        )

        # state space, [temperature, carbon emissions, carbon concentration, radiative forcing]
        if not multigas:
            self.observation_space = gym.spaces.Box(
                np.array([-100, -100, 0, -100]).astype(np.float32),
                np.array([100, 100, 5000, 100]).astype(np.float32),
            )
        else:
            self.observation_space = gym.spaces.Box(
                np.array([-100, -100, 0, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50]).astype(np.float32),
                np.array([100, 100, 5000, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]).astype(np.float32),
            )

        # specify the reward function to use
        if reward_mode == "simple":
            self.reward_func = simple_reward
        elif reward_mode == "temp":
            self.reward_func = temp_reward
        elif reward_mode == "conc":
            self.reward_func = conc_reward
        elif reward_mode == "carbon_cost":
            self.reward_func = carbon_cost_reward
        elif reward_mode == "temp_emit":
            self.reward_func = temp_emit_reward
        elif reward_mode == "temp_emit_diff":
            self.reward_func = temp_emit_diff_reward

        # setup additional forcing factors
        if forcing:
            solar = 0.1 * np.sin(2 * np.pi * np.arange(736) / 11.5)
            volcanic = -gamma.rvs(0.2, size=736, random_state=14)
            self.forward_func = partial(fair.forward.fair_scm, F_solar=solar, F_volcanic=volcanic)
        else:
            self.forward_func = fair.forward.fair_scm
        self.multigas = multigas
        self.scenario = scenario

        # set the initial state
        self.reset()

    def update_state(self, C, F, T):
        # state is temp, emissions, concentration, forcing factors
        concentration = C[256+self.t]
        forcing = F[256+self.t]
        if self.multigas:
            concentration = concentration[0]
            forcing = list(forcing)
        else:
            forcing = [forcing]
        self.state = [T[256+self.t], self.emissions[256+self.t], concentration] + forcing
        self.t += 1
    
    def reset(self):
        # initialize historical emissions fro RCP scenario
        base_emissions = eval(self.scenario).Emissions.emissions
        if not self.multigas:
            base_emissions = np.array([x[1]+x[2] for x in base_emissions])
        # 80 year time horizon, meet goals by 2100
        self.emissions = base_emissions
        # 2021 estimate of GtC of carbon emissions
        # 2021 is the 257th year in the rcp scenario
        if not self.multigas:
            self.emissions[256] = 36
        else:
            self.emissions[256][1] = 36*.9
            self.emissions[256][2] = 36*.1
        self.t = 0
        # initial state
        C, F, T = self.forward_func(
            emissions=self.emissions,
            useMultigas=self.multigas
        )
        self.update_state(C, F, T)

        return self.state

    def step(self, action):
        done = False
        
        # change emissions by the action amount
        if not self.multigas:
            self.emissions[256+self.t] = max(self.emissions[256+self.t-1] + action[0], 0)
        else:
            self.emissions[256+self.t][1] = max(self.emissions[256+self.t-1][1] + action[0]*.9, 0)
            self.emissions[256+self.t][2] = max(self.emissions[256+self.t-1][2] + action[0]*.1, 0)

        # run FaIR simulator
        C, F, T = self.forward_func(
            emissions=self.emissions,
            useMultigas=self.multigas
        )
        
        # fail if temperature error
        if math.isnan(T[256+self.t]):
            done = True

        # compute the reward
        cur_emit = self.emissions[256+self.t]
        cur_conc = C[256+self.t]    
        if self.multigas:
            cur_emit = cur_emit[1] + cur_emit[2]
            cur_conc = cur_conc[0]
        reward = self.reward_func(self.state, T[256+self.t], self.t, cur_emit, cur_conc)
            
        # update the state and info
        self.update_state(C, F, T)

        # end the trial once 2100 is reached
        if self.t == 79 or self.state[0] > 2:
            done = True

        return self.state, reward, done, {}

    def render(self, mode="human"):
        # print the state
        print(f'Temperature anomaly: {self.state[0]}ºC')
        print(f'CO2 emissions: {self.state[1]} GtC')
        print(f'CO2 concentration: {self.state[2]} ppm')
        print(f'Radiative forcing: {self.state[3:]}')


#### Training code ####

# Output useful plots
def make_plots(vals, args, save_path):
    plots = MULTIGAS_VARS if args.multigas else VARS
    for i in range(len(plots)):
        name = plots[i][:plots[i].find('(')].strip()
        ys = [x[i] for x in vals]
        xs = [2021+x for x in range(len(vals))]
        plt.plot(xs, ys)
        plt.ylabel(plots[i])
        plt.xlabel('Year')
        plt.savefig(os.path.join(save_path, 'plots', name))
        plt.clf()


# Main training and eval loop
def main(args, save_path):
    # Create the environment
    env = Simulator(
        action_space = args.action_space,
        reward_mode = args.reward_mode,
        forcing = args.forcing,
        multigas = args.multigas,
        scenario = args.scenario
    )

    # Train the algorithm
    if args.algorithm == 'a2c':
        model_builder = A2C
    elif args.algorithm == 'ppo':
        model_builder = PPO
    elif args.algorithm == 'ddpg':
        model_builder = DDPG
    model = model_builder(
            policy="MlpPolicy",
            env=env, 
            learning_rate = args.lr,
            n_steps = args.n_steps,
            gamma=args.gamma,
            verbose=1,
            device=args.device,
            tensorboard_log=os.path.join(save_path, 'logs'),
    )
    model.learn(
        total_timesteps=args.timesteps,
        eval_freq=20,
        log_interval=20,
        eval_log_path=os.path.join(save_path, 'evals')
    )
    model.save(f'{save_path}/model_state_dict.pt')

    # Run evaluation and make plots
    obs = env.reset()
    vals = []
    for i in range(80):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        vals.append(obs + [reward])
        if done:
          break
    env.close()
    make_plots(vals, args, save_path)


# Make output directories
def make_outdirs(save_path):
    dirs = ['plots', 'logs']
    for dir in dirs:
        path = os.path.join(save_path, dir)
        if not os.path.exists(path):
            os.makedirs(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='test', required=False)
    parser.add_argument("--action_space", type=int, default=2, required=False)
    parser.add_argument("--reward_mode", type=str, default='simple', required=False) # 'simple', 'temp', 'conc', 'carbon_cost', temp_emit', 'temp_emit_diff'
    parser.add_argument("--forcing", action='store_true', required=False)
    parser.add_argument("--output_path", type=str, default='outputs', required=False)
    parser.add_argument("--stdout", action='store_true', required=False)
    parser.add_argument("--seed", type=int, default=random.randint(1,1000), required=False)
    parser.add_argument("--device", type=str, default='cpu', required=False)
    parser.add_argument("--lr", type=float, default=7e-4, required=False)
    parser.add_argument("--n_steps", type=int, default=5, required=False)
    parser.add_argument("--gamma", type=float, default=0.99, required=False)
    parser.add_argument("--timesteps", type=int, default=10000, required=False)
    parser.add_argument("--algorithm", type=str, default='a2c', required=False)
    parser.add_argument("--multigas", action='store_true', required=False)
    parser.add_argument("--scenario", type=str, default='rcp60', required=False)
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup save path and logging 
    save_path = os.path.join(args.output_path, args.name)
    make_outdirs(save_path)
    start_time = time.time()
    if not args.stdout:
        sys.stdout = open(os.path.join(save_path, 'stdout.txt'), 'w')
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args, save_path)

    # log total runtime and close logging file
    print(f'\nTOTAL RUNTIME: {int((time.time() - start_time)/60.)} minutes {int((time.time() - start_time) % 60)} seconds')
    if not args.stdout:
        sys.stdout.close()

