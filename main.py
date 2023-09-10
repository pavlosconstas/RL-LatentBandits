import numpy as np
from hmmlearn import hmm
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import matplotlib.pyplot as plt


class Patient:
    def __init__(self) -> None:
        self.depression = np.random.randint(1, 6)
        self.anxiety = np.random.randint(1, 6)
        self.insomnia = np.random.randint(1, 6)
        self.regular_speech_fluency = np.clip(np.random.normal(0.75, 0.1), 0, 1)
        self.medication_accumulation = []
        self.side_effect_accumulation = 0

        self.depression_model = self.generate_model(self.depression)
        self.anxiety_model = self.generate_model(self.anxiety)
        self.insomnia_model = self.generate_model(self.insomnia)

        self.current_depression_state = self.depression
        self.current_anxiety_state = self.anxiety
        self.current_insomnia_state = self.insomnia
        self.current_speech_fluency = self.calculate_speech_fluency()

    def generate_model(self, initial_state):
        transmat_ = self.generate_transition_matrix(initial_state)

        model = hmm.GaussianHMM(n_components=5, covariance_type="full")
        model.startprob_ = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        model.transmat_ = transmat_
        model.means_ = np.array([[0.8], [0.7], [0.6], [0.5], [0.4]])
        model.covars_ = np.tile(np.array([[0.01]]), (5, 1, 1))
        model.n_features = 1

        return model

    def generate_transition_matrix(self, initial_state):
        if initial_state == 1:
            return np.array([
                [0.9, 0.1, 0.0, 0.0, 0.0],
                [0.2, 0.7, 0.1, 0.0, 0.0],
                [0.2, 0.3, 0.3, 0.2, 0.0],
                [0.1, 0.2, 0.3, 0.3, 0.1],
                [0.2, 0.2, 0.2, 0.2, 0.2]
            ])
        elif initial_state == 2:
            return np.array([
                [0.3, 0.6, 0.1, 0.0, 0.0],
                [0.1, 0.8, 0.1, 0.0, 0.0],
                [0.0, 0.3, 0.6, 0.1, 0.0],
                [0.0, 0.1, 0.3, 0.5, 0.1],
                [0.1, 0.1, 0.2, 0.3, 0.3]
            ])

        elif initial_state == 3:
            return np.array([
                [0.1, 0.1, 0.7, 0.1, 0.0],
                [0.1, 0.1, 0.7, 0.1, 0.0],
                [0.0, 0.1, 0.8, 0.1, 0.0],
                [0.0, 0.1, 0.6, 0.2, 0.1],
                [0.0, 0.1, 0.4, 0.3, 0.2]
            ])
        elif initial_state == 4:
            return np.array([
                [0.1, 0.0, 0.1, 0.7, 0.1],
                [0.1, 0.0, 0.1, 0.7, 0.1],
                [0.0, 0.1, 0.1, 0.7, 0.1],
                [0.0, 0.0, 0.1, 0.8, 0.1],
                [0.0, 0.0, 0.1, 0.6, 0.3]
            ])
        elif initial_state == 5:
            return np.array([
                [0.1, 0.1, 0.0, 0.1, 0.7],
                [0.1, 0.1, 0.0, 0.1, 0.7],
                [0.1, 0.1, 0.0, 0.1, 0.7],
                [0.0, 0.1, 0.1, 0.1, 0.7], 
                [0.0, 0.0, 0.1, 0.1, 0.8]
            ])
    
    def calculate_speech_fluency(self):
        # Calculate the inverse scores for each condition
        depression_score = (6 - self.current_depression_state) / 5
        anxiety_score = (6 - self.current_anxiety_state) / 5
        insomnia_score = (6 - self.current_insomnia_state) / 5
        
        # Calculate the weighted average
        speech_fluency = (0.1 * depression_score + 0.2 * anxiety_score + 0.3 * insomnia_score + 0.4 * self.regular_speech_fluency)

        return speech_fluency

    def generate_one_step(self):
        self.current_depression_state = np.random.choice(np.arange(len(self.depression_model.transmat_)), p=self.depression_model.transmat_[self.current_depression_state - 1]) + 1
        self.current_anxiety_state = np.random.choice(np.arange(len(self.anxiety_model.transmat_)), p=self.anxiety_model.transmat_[self.current_anxiety_state - 1]) + 1
        self.current_insomnia_state = np.random.choice(np.arange(len(self.insomnia_model.transmat_)), p=self.insomnia_model.transmat_[self.current_insomnia_state - 1]) + 1

        current_depression_obs = np.random.normal(self.depression_model.means_[self.current_depression_state - 1], np.sqrt(self.depression_model.covars_[self.current_depression_state - 1]))
        current_anxiety_obs = np.random.normal(self.anxiety_model.means_[self.current_anxiety_state - 1], np.sqrt(self.anxiety_model.covars_[self.current_anxiety_state - 1]))
        current_insomnia_obs = np.random.normal(self.insomnia_model.means_[self.current_insomnia_state - 1], np.sqrt(self.insomnia_model.covars_[self.current_insomnia_state - 1]))

        self.current_speech_fluency = self.calculate_speech_fluency()

        return current_depression_obs[0][0], current_anxiety_obs[0][0], current_insomnia_obs[0][0]
    
    def administer_medication(self, medication, current_day):

        medication.day_administered = current_day
        self.medication_accumulation.append(medication)

    def update_medication_dosage(self, current_day):
        for medication in self.medication_accumulation:
            medication.update_dosage(current_day)
        
        self.medication_accumulation = [med for med in self.medication_accumulation if med.dosage > 0.0005]


    def adjust_transition_matrix(self, matrix, effect):
        if effect > 0:
            for i in range(len(matrix)):
                if i != 0:
                    matrix[i][i-1] = max(matrix[i][i-1] + effect, 0)  # increase probability of improving
                matrix[i][i] = max(matrix[i][i] + effect, 0)  # increase probability of staying the same
                if i != len(matrix)-1:
                    matrix[i][i+1] = max(matrix[i][i+1] - effect, 0)  # decrease probability of worsening

                # Normalize the probabilities so they sum to 1
                matrix[i] = matrix[i] / np.sum(matrix[i])
        return matrix



    def apply_medication_effects(self, current_day):
        for medication in self.medication_accumulation:
            medication.check_if_is_active(current_day)
            if medication.is_active:
                depression_effect = medication.depression_effect[0] * medication.dosage + np.random.uniform(-medication.depression_effect[1], medication.depression_effect[1])
                anxiety_effect = medication.anxiety_effect[0] * medication.dosage + np.random.uniform(-medication.anxiety_effect[1], medication.anxiety_effect[1])
                insomnia_effect = medication.insomnia_effect[0] * medication.dosage + np.random.uniform(-medication.insomnia_effect[1], medication.insomnia_effect[1])
                
                # Adjust transition matrices based on medication effects
                self.depression_model.transmat_ = self.adjust_transition_matrix(self.depression_model.transmat_, depression_effect)
                self.anxiety_model.transmat_ = self.adjust_transition_matrix(self.anxiety_model.transmat_, anxiety_effect)
                self.insomnia_model.transmat_ = self.adjust_transition_matrix(self.insomnia_model.transmat_, insomnia_effect)
                
                # Apply side effects
                self.side_effect_accumulation += medication.side_effect[0] * medication.dosage + np.random.uniform(-medication.side_effect[1], medication.side_effect[1])

        # Update speech fluency
        self.current_speech_fluency = self.calculate_speech_fluency()





    def __str__(self) -> str:
        return f"Depression: {self.depression}\nAnxiety: {self.anxiety}\nInsomnia: {self.insomnia}\nRegular Speech Fluency: {self.regular_speech_fluency}\n"

class Medication:
    def __init__(self, name: str, depression_effect: (float, float), anxiety_effect: (float, float), insomnia_effect: (float, float), side_effect: (float, float), dosage: int, time_to_effect: int, half_life: int) -> None:
        self.name = name
        self.depression_effect = depression_effect
        self.anxiety_effect = anxiety_effect
        self.insomnia_effect = insomnia_effect
        self.side_effect = side_effect
        self.dosage = dosage
        self.time_to_effect = time_to_effect
        self.half_life = half_life
        self.day_administered = 0
        self.is_active = False
    
    def check_if_is_active(self, current_day):
        self.is_active = current_day >= self.day_administered + self.time_to_effect
    
    def update_dosage(self, current_day):
        days_since_administered = current_day - self.day_administered

        if days_since_administered % self.half_life == 0:
            self.dosage /= 2

    def __str__(self) -> str:
        return f"Name: {self.name}\nDepression Effect: {self.depression_effect}\nAnxiety Effect: {self.anxiety_effect}\nInsomnia Effect: {self.insomnia_effect}\nSide Effect: {self.side_effect}\nTime to Effect: {self.time_to_effect}\nHalf Life: {self.half_life}\n"



med1 = Medication(name="Med1", depression_effect=(0.1, 0.1), anxiety_effect=(0, 0), insomnia_effect=(0, 0), side_effect=(0.05, 0.1), dosage=1, time_to_effect=14, half_life=3)
med2 = Medication(name="Med2", depression_effect=(0, 0), anxiety_effect=(0.1, 0.1), insomnia_effect=(0, 0), side_effect=(0.05, 0.1), dosage=1, time_to_effect=21, half_life=4)
med3 = Medication(name="Med3", depression_effect=(0.05, 0.05), anxiety_effect=(0.1, 0.1), insomnia_effect=(-0.1, 0.1), side_effect=(0.03, 0.08), dosage=1, time_to_effect=10, half_life=5)
med4 = Medication(name="Med4", depression_effect=(0.2, 0.1), anxiety_effect=(-0.05, 0.05), insomnia_effect=(0.1, 0.1), side_effect=(0.1, 0.15), dosage=1, time_to_effect=7, half_life=6)
med5 = Medication(name="Med5", depression_effect=(0, 0), anxiety_effect=(0.2, 0.1), insomnia_effect=(0, 0), side_effect=(0.02, 0.07), dosage=1, time_to_effect=12, half_life=4)
med6 = Medication(name="Med6", depression_effect=(-0.05, 0.05), anxiety_effect=(0, 0), insomnia_effect=(0.15, 0.1), side_effect=(0.06, 0.11), dosage=1, time_to_effect=20, half_life=3)
med7 = Medication(name="Med7", depression_effect=(0.15, 0.1), anxiety_effect=(0.15, 0.1), insomnia_effect=(0, 0), side_effect=(0.07, 0.12), dosage=1, time_to_effect=15, half_life=5)
med8 = Medication(name="Med8", depression_effect=(0, 0), anxiety_effect=(-0.1, 0.1), insomnia_effect=(0.05, 0.05), side_effect=(0.04, 0.09), dosage=1, time_to_effect=18, half_life=7)
med9 = Medication(name="Med9", depression_effect=(0.05, 0.05), anxiety_effect=(0.05, 0.05), insomnia_effect=(0.05, 0.05), side_effect=(0.01, 0.05), dosage=1, time_to_effect=14, half_life=4)
med10 = Medication(name="Med10", depression_effect=(0.1, 0.1), anxiety_effect=(0.1, 0.1), insomnia_effect=(-0.05, 0.05), side_effect=(0.08, 0.13), dosage=1, time_to_effect=16, half_life=6)

# You can add more medications here

class MedicationEnvironment(py_environment.PyEnvironment):
    
    def __init__(self, patient, medication_list):
        self._patient = patient
        self._medication_list = medication_list
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=len(self._medication_list) - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(3,), dtype=np.float32, minimum=[0, 0, 0], maximum=[1, 1, 1], name='observation')
        self._state = np.array([self._patient.depression, self._patient.anxiety, self._patient.insomnia], dtype=np.float32)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._state = np.array([self._patient.depression, self._patient.anxiety, self._patient.insomnia], dtype=np.float32)
        self._episode_ended = False
        return ts.restart(self._state)
    
    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._patient.administer_medication(self._medication_list[action], 0)
        depression_X, anxiety_X, insomnia_X = self._patient.generate_one_step()
        self._state = np.array([depression_X, anxiety_X, insomnia_X], dtype=np.float32)

        print(f"Depression Level: {depression_X}, Anxiety Level: {anxiety_X}, Insomnia Level: {insomnia_X}")

        # You can define a reward mechanism here based on the improvements or deteriorations in health conditions.
        reward = np.sum(self._state)  # Negative sum of conditions as we want to minimize them

        print(f"Reward: {reward}")

        if np.sum(self._state) <= 0.5:
            self._episode_ended = True
            print("Episode Ended: Patient's health conditions have significantly improved.")
            return ts.termination(self._state, reward)

        return ts.transition(self._state, reward=reward, discount=1.0)


patient = Patient()

# Convert our environment to a TensorFlow environment
train_env = tf_py_environment.TFPyEnvironment(MedicationEnvironment(patient, [med1, med2, med3, med4, med5, med6, med7, med8, med9, med10]))
eval_env = tf_py_environment.TFPyEnvironment(MedicationEnvironment(patient, [med1, med2, med3, med4, med5, med6, med7, med8, med9, med10]))

# Initialize the LinUCB Agent
agent = lin_ucb_agent.LinearUCBAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    tikhonov_weight=1.0,
    alpha=10.0,
    dtype=tf.float32)


# Metrics and Evaluation
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=2000)

metrics = [tf_metrics.AverageReturnMetric()]

# Driver
collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch] + metrics,
    num_steps=1)

# Training Loop
raw_rewards = []
average_returns = []

num_iterations = 1000
for i in range(num_iterations):
    collect_driver.run()
    trajectories = replay_buffer.gather_all()
    rewards = trajectories.reward.numpy()
    raw_rewards.extend(rewards)
    replay_buffer.clear()
    average_return = sum(raw_rewards) / len(raw_rewards)
    average_returns.append(average_return)
    if i % 50 == 0:
        print(f"Iteration {i}, Average Return: {average_return}")


plt.figure(figsize=(10, 5))
plt.plot(average_returns)
plt.xlabel("Iterations")
plt.ylabel("Average Return")
plt.title("Average Return over Time")
plt.grid(True)
plt.show()

# Evaluate the agent's policy
# print('Average Return:', metrics[0].result().numpy())