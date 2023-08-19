import numpy as np
import math




def prob_X_given_Z(X, Z_states):

  mean = 1 - (0.1 * Z_states['anxiety_level']) - (0.1 * Z_states['depression_level']) - (0.1 * Z_states['insomnia_level'])

  standard_deviation = 0.1

  [[MA, SA],
   [MD, SD],
   [MI, SI]]

  return (1 / (standard_deviation * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((X - mean) / standard_deviation) ** 2), mean, standard_deviation

def prob_R_given_Z_A(R, Z_states, action):
  
    mean = 1 - (0.1 * Z_states['anxiety_level']) - (0.1 * Z_states['depression_level']) - (0.1 * Z_states['insomnia_level'])
  
    if action == 'm1':
      mean = mean + 0.1
  
    standard_deviation = 0.1
  
    return (1 / (standard_deviation * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((R - mean) / standard_deviation) ** 2)

# function builds the inputs for the mining example (Gaussian version) of a non stationary latent bandit
# the actions are: heavy and light mining
def inputs_mining_exp():

  '''
  # parameters for P(X|Z)
  beta_0 = -0.18 # relationship between X and Z | y = mx + b | P(X) = beta_1 * z + beta_0
  beta_1 = 1.32 # relationship between X and Z
  sigma = 0.62 # standard deviation of the context (inherent variability)
  tau = 0.45 #  standard deviation of the context (variability of the latent state)
  '''

  # parameters for P(R|A,Z)

  Z_states = {
    'anxiety_level': (1, 2, 3, 4, 5),          # (0, 0, 1, 0, 0) (0.1, 0.2, 0.3, 0.3, 0.1)
    'depression_level': (1, 2, 3, 4, 5),       # (0, 0, 0, 1, 0)
    'insomnia_level': (1, 2, 3, 4, 5),         # (1, 0, 0, 0, 0)
    'ssri_accum': 0,
    'z-drug_accum': 0
  }

  A_states = {
    'm1': {
      'depression_effect': 0.4,
      'anxiety_effect': 0.1,
      'insomnia_effect': 0.3,
      'z-drug_dose': 0.5,
      'ssri_dose': 0.3,
      'cost': 0.5,
      'effect_time': 12
    }
  }

  # Z_states = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1] # depression, anxiety, insomnia
  # phi_star_fun(Z_states)
  '''
  A_states = [{
    'reference': 'm1',
    'reward_factor': 0.1,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm2',
    'reward_factor': 0.2,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm3',
    'reward_factor': 0.3,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm4',
    'reward_factor': 0.4,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm5',
    'reward_factor': 0.5,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm6',
    'reward_factor': 0.6,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm7',
    'reward_factor': 0.7,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm8',
    'reward_factor': 0.8,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm9',
    'reward_factor': 0.9,
    'cost': 1,
    'var_cost_sigma': 0.5
  }, {
    'reference': 'm10',
    'reward_factor': 1,
    'cost': 1,
    'var_cost_sigma': 0.5
  }]  # heavy mining vs. light mining actions

  '''

  '''
  # P(X|Z)
  mean_X_given_Z = [beta_0 + (beta_1 * Z) for Z in Z_states]
  stdev_X_given_Z = [math.sqrt((sigma**2) + (tau**2))] * len(Z_states)

  # P(R|Z,A)

  mean_R_given_Z_action = [[(mu * A["reward_factor"]) - A["cost"] for mu in mean_X_given_Z] for A in A_states]
  stdev_R_given_Z_action = [(math.sqrt((A["reward_factor"] * (sigma**2)) + (A["cost"] ** 2))) for A in A_states]
  '''
  # latent transition matrix = P(Z'|Z)
  # phi_star = 

  # Z, K = len(Z_states), len(A_states)
  
  # # convert to numpy arrays
  # theta_star = np.transpose(np.array([mean_X_given_Z, stdev_X_given_Z]))
  # reward_params = np.empty((Z, K, 2))

  # for i in range(0, 1):
  #   if (i == 0):
  #     for j in range(0, K):
  #       reward_params[:, j, i] = mean_R_given_Z_action[j]
  #   else:
  #     for j in range(0, K):
  #       reward_params[:, j, i] = stdev_R_given_Z_action[j]


  return prob_X_given_Z, prob_R_given_Z_A, Z_states, A_states
