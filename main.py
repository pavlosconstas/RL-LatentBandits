import numpy as np
#import random
import math
import copy
from env import DynamicConfounderBanditGaussian
import env_params_dosing
from algorithm import run
from linear_bandit import run_linear
from cmd_args import cmd_args


#random.seed(0)
np.random.seed(0)


env_type = 'gaussian'
#env_type = 'discrete'

if env_type=='gaussian':

    # mining application parameters
    theta_star, phi_star, reward_params = env_params_dosing.inputs_mining_exp()

    Z = phi_star.shape[0]
    probs_latent_init = np.random.rand(Z,) # np.ones(Z,)/Z
    probs_latent_init /= probs_latent_init.sum()

    env = DynamicConfounderBanditGaussian(theta_star, probs_latent_init, phi_star, reward_params)

ep_length = 10000
num_episodes = 100

#run_linear(env, ep_length, num_episodes)

bandit_algo = 'ts'
#bandit_algo = 'ucb'
oracle_posterior=False
#oracle_posterior=True
offline_samples=False
run(env, ep_length, num_episodes, oracle_posterior=oracle_posterior, offline_samples=offline_samples, algo=bandit_algo)

