import numpy as np
import matplotlib.pyplot as plt

from algorithm import LinUCB
from env import Medication as Md, SimulationEnvironment as Se
from hmm import PatientWithHMM as Phmm, Patient as Pt


def run_simulation_with_linucb(environment, alpha, d, num_iterations):
    agent = LinUCB(alpha, d)
    total_reward = 0
    rewards = []

    '''
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
        '''
    
    for _ in range(num_iterations):
        predicted_rewards = []

        for med in environment.medications:

            x = np.array([
                environment.patient.speech_fluency,
                med.effect_depression[0],
                med.effect_anxiety[0],
                med.effect_insomnia[0],
                med.side_effect[0]
            ])
            predicted_reward = agent.predict(x)
            predicted_rewards.append(predicted_reward)
        
        chosen_medication = environment.medications[np.argmax(predicted_rewards)]

        chosen_medications = [environment.medications[x] for x in np.argsort(predicted_rewards)[-5:]]

        # reward = environment.step(chosen_medication)

        reward = [environment.step(chosen_medication) for chosen_medication in chosen_medications]

        # chosen_x = np.array([
        #     environment.patient.speech_fluency,
        #     chosen_medication.effect_depression[0],
        #     chosen_medication.effect_anxiety[0],
        #     chosen_medication.effect_insomnia[0],
        #     chosen_medication.side_effect[0]
        # ])
        
        chosen_x = [np.array([
            environment.patient.speech_fluency,
            chosen_medication.effect_depression[0],
            chosen_medication.effect_anxiety[0],
            chosen_medication.effect_insomnia[0],
            chosen_medication.side_effect[0]
        ]) for chosen_medication in chosen_medications]

        agent.update(chosen_x, reward)

        total_reward += sum(reward)
        rewards.append(total_reward)

    return rewards

medication_a = Md("anxiety1", (0.2, 0.1), (-0.3, 0.1), (-0.2, 0.1), (0.1, 0.05), 4)
medication_b = Md("anxiety2", (-0.2, 0.1), (-0.3, 0.1), (0.2, 0.1), (0.1, 0.05), 3)
medication_c = Md("z-drug", (0.0, 0.1), (0.1, 0.1), (-0.4, 0.1), (0.1, 0.05), 4)
medication_d = Md("anti-depression", (-0.4, 0.1), (-0.1, 0.1), (0.1, 0.1), (0.1, 0.05), 7)
medication_e = Md("anti-depression2", (-0.3, 0.1), (0.1, 0.1), (-0.1, 0.1), (0.1, 0.05), 4)


simulation = Se([medication_a, medication_b, medication_c, medication_d, medication_e])
reward_med_a = simulation.step(medication_a)
reward_med_b = simulation.step(medication_b)
reward_med_b = simulation.step(medication_c)
reward_med_b = simulation.step(medication_d)
reward_med_b = simulation.step(medication_e)

reward_med_a, reward_med_b, medication_c, medication_d, medication_e

# Test the LinUCB with the simulation environment
rewards = run_simulation_with_linucb(simulation, alpha=1, d=5, num_iterations=30)
rewards

def run_multiple_simulations(num_simulations=10, num_iterations=30):
    all_rewards = []

    for _ in range(num_simulations):
        simulation = Se([medication_a, medication_b, medication_c, medication_d, medication_e])
        rewards = run_simulation_with_linucb(simulation, alpha=1, d=5, num_iterations=num_iterations)
        all_rewards.append(rewards)

    return all_rewards

num_simulations = 20
all_rewards = run_multiple_simulations(num_simulations=num_simulations)

# Plotting the results
plt.figure(figsize=(10, 6))
for i in range(num_simulations):
    plt.plot(all_rewards[i], alpha=0.5)
plt.xlabel('Iterations')
plt.ylabel('Cumulative Reward')
plt.title('Reward over Time across Multiple Simulations')
plt.show()


'''
reward_at_i = [np.diff([0] + r) for r in all_rewards]

average_rewards = np.mean(reward_at_i, axis=0)

plt.plot(average_rewards)
plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title('Reward over Time across Multiple Simulations')
plt.show()
'''