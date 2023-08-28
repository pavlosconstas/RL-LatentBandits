import numpy as np
from hmm import PatientWithHMM

class Medication:
    def __init__(self, name, effect_depression, effect_anxiety, effect_insomnia, side_effect, time_to_effect):
        self.name = name
        self.effect_depression = effect_depression # tuple(mean, std_dev)
        self.effect_anxiety = effect_anxiety
        self.effect_insomnia = effect_insomnia
        self.side_effect = side_effect
        self.time_to_effect = time_to_effect
        self.current_effect_time = 0

    def apply_effect(self):
        if self.current_effect_time < self.time_to_effect:
            self.current_effect_time += 1
            return (0, 0, 0, 0)
        else:
            return (
                np.random.normal(self.effect_depression[0], self.effect_depression[1]),
                np.random.normal(self.effect_anxiety[0], self.effect_anxiety[1]),
                np.random.normal(self.effect_insomnia[0], self.effect_insomnia[1]),
                np.random.normal(self.side_effect[0], self.side_effect[1])
            )


class SimulationEnvironment:
    def __init__(self, medications, initial_states=(3, 2, 4), time_periods=30):
        self.medications = medications
        self.patient = PatientWithHMM(*initial_states)
        self.time_periods = time_periods
        self.current_time = 0
        self.reward_history = []

    def reset(self):
        self.patient = PatientWithHMM(*self.initial_states)
        self.current_time = 0
        self.reward_history.clear()

    def calculate_reward(self):
        # Reward is based on speech fluency (higher is better) and side effects (lower is better)
        # For simplicity, I am using a linear combination here. This can be refined as per requirements.
        return self.patient.speech_fluency - 0.5 * (6 - (self.patient.insomnia + self.patient.depression + self.patient.anxiety))

    def step(self, medication):
        self.patient.take_medication(medication)
        for _ in range(medication.time_to_effect):
            self.patient.update_states()
            self.current_time += 1
            reward = self.calculate_reward()
            self.reward_history.append(reward)
            if self.current_time >= self.time_periods:
                return sum(self.reward_history)

        return self.calculate_reward()