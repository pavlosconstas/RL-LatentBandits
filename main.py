import numpy as np
import matplotlib.pyplot as plt

from algorithm import LinUCB
from env import Medication as Md, SimulationEnvironment as Se
from hmm import PatientWithHMM as Phmm, Patient as Pt


def run_simulation_with_linucb(environment, alpha, d, num_iterations):
    agent = LinUCB(alpha, d)
    total_reward = 0
    rewards = []

    for _ in range(num_iterations):
        # For simplicity, we'll use the patient's speech fluency as the feature vector
        x = np.array([environment.patient.speech_fluency])
        
        # Predict rewards for each medication
        predicted_rewards = [agent.predict(x) for med in environment.medications]
        
        # Choose the medication with the highest predicted reward
        chosen_medication = environment.medications[np.argmax(predicted_rewards)]
        
        # Take the step in the environment with the chosen medication
        reward = environment.step(chosen_medication)
        total_reward += reward
        rewards.append(total_reward)
        
        # Update the agent
        agent.update(x, reward)

    return rewards

medication_example = Md("MedA", (0.2, 0.1), (-0.3, 0.1), (-0.2, 0.1), (0.1, 0.05), 2)
patient_example = Pt(3, 2, 4)
patient_example.take_medication(medication_example)
patient_example.update_states()

patient_example.speech_fluency


patient_hmm_example = Phmm(3, 2, 4)
patient_hmm_example.take_medication(medication_example)
patient_hmm_example.update_states()

patient_hmm_example.speech_fluency, patient_hmm_example.insomnia, patient_hmm_example.depression, patient_hmm_example.anxiety


medication_b = Md("MedB", (-0.2, 0.1), (-0.3, 0.1), (0.2, 0.1), (0.1, 0.05), 3)
simulation = Se([medication_example, medication_b])
reward_med_a = simulation.step(medication_example)
reward_med_b = simulation.step(medication_b)

reward_med_a, reward_med_b

# Test the LinUCB with the simulation environment
rewards = run_simulation_with_linucb(simulation, alpha=1, d=1, num_iterations=30)
rewards

def run_multiple_simulations(num_simulations=1000, num_iterations=50):
    all_rewards = []

    for _ in range(num_simulations):
        simulation = Se([medication_example, medication_b])
        rewards = run_simulation_with_linucb(simulation, alpha=1, d=1, num_iterations=num_iterations)
        all_rewards.append(rewards)

    return all_rewards

num_simulations = 100
all_rewards = run_multiple_simulations(num_simulations=num_simulations)

# Plotting the results
plt.figure(figsize=(10, 6))
for i in range(num_simulations):
    plt.plot(all_rewards[i], alpha=0.5)
plt.xlabel('Iterations')
plt.ylabel('Cumulative Reward')
plt.title('Reward over Time across Multiple Simulations')
plt.show()
