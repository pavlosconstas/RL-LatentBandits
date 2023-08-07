import numpy as np
import math


# function builds the inputs for the mining example (Gaussian version) of a non stationary latent bandit
# the actions are: heavy and light mining
def inputs_mining_exp():

  # parameters for P(X|Z)
  beta_0 = -0.18 # relationship between X and Z | y = mx + b | P(X) = beta_1 * z + beta_0
  beta_1 = 1.32 # relationship between X and Z
  sigma = 0.62 # standard deviation of the context (inherent variability)
  tau = 0.45 #  standard deviation of the context (variability of the latent state)

  # parameters for P(R|A,Z)

  Z_states = [1, 2, 5] # depression, anxiety, insomnia

  A_states = [{
    'reference': 'm1',
    'reward_factor': 0.1,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm2',
    'reward_factor': 0.1,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm3',
    'reward_factor': 0.1,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm4',
    'reward_factor': 0.1,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm5',
    'reward_factor': 0.1,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm6',
    'reward_factor': 0.1,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm7',
    'reward_factor': 0.1,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm8',
    'reward_factor': 0.1,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm9',
    'reward_factor': 0.1,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm10',
    'reward_factor': 0.1,
    'cost': 1,
    'var_cost_sigma': 0.5
  }]  # heavy mining vs. light mining actions

  # P(X|Z)
  mean_X_given_Z = [beta_0 + (beta_1 * Z) for Z in Z_states]
  stdev_X_given_Z = [math.sqrt((sigma**2) + (tau**2))] * len(Z_states)

  # P(R|Z,A)

  mean_R_given_Z_action = [[(mu * A["reward_factor"]) - A["cost"] for mu in mean_X_given_Z] for A in A_states]
  stdev_R_given_Z_action = [(math.sqrt((A["reward_factor"] * (sigma**2)) + (A["cost"] ** 2))) for A in A_states]

  # latent transition matrix = P(Z'|Z)
  phi_star = np.array([[0.7,0.25,0.05],[0.25,0.5,0.25],[0.05,0.25,0.7]])


  Z, K = len(Z_states), len(A_states)
  
  # convert to numpy arrays
  theta_star = np.transpose(np.array([mean_X_given_Z, stdev_X_given_Z]))
  reward_params = np.empty((Z, K, 2))

  for i in range(0, Z-1):
    if (i == 0):
      for j in range(0, K):
        reward_params[:, j, i] = mean_R_given_Z_action[j]
    else:
      for j in range(0, K):
        reward_params[:, j, i] = stdev_R_given_Z_action[j]


  return theta_star, phi_star, reward_params
