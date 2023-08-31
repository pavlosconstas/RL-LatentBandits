import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from hmm import Patient
from env import Medication, SimulationEnvironment
from algorithm import LinUCB

# Step 4: Running the Simulation

def run_simulation(alpha, num_days, medications):
    patient = Patient()
    env = SimulationEnvironment(patient, medications)
    agent = LinUCB(alpha=alpha, num_arms=len(medications), d=4)
    
    total_rewards = []
    cumulative_reward = 0
    
    for day in range(num_days):
        context = np.array([env.patient.speech_fluency, env.patient.hours_slept_previous_night, env.patient.medication_accumulation, day])
        action = agent.choose_action(context)
        context, reward = env.step(action)
        agent.update(action, reward, context)
        
        cumulative_reward += reward
        total_rewards.append(cumulative_reward)
    
    return total_rewards

# Testing the Simulation
test_medications = [Medication("Med" + str(i), (0, 1), (0, 1), (0, 1), (0, 1), 2) for i in range(10)]
rewards = run_simulation(alpha=1, num_days=10, medications=test_medications)


def run_optimized_simulation(alpha_values, num_days, medications):
    results = {}
    for alpha in alpha_values:
        results[alpha] = run_simulation(alpha, num_days, medications)
    return results

def analyze_results(results):
    analysis_data = {}
    for alpha, rewards in results.items():
        slope, _, _, p_value, _ = linregress(range(len(rewards)), rewards)
        analysis_data[alpha] = {
            'slope': slope,
            'p_value': p_value
        }
    return analysis_data

# Running the optimized simulation
alpha_values = [0.1, 0.5, 1, 1.5, 2]
results = run_optimized_simulation(alpha_values, num_days=30, medications=test_medications)

# Visualization
plt.figure(figsize=(12, 6))
for alpha, rewards in results.items():
    plt.plot(rewards, label=f"alpha = {alpha}")

plt.title("Cumulative Reward over Time for Different Alpha Values")
plt.xlabel("Days")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)
plt.show()