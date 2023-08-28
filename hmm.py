import numpy as np


class HiddenMarkovModel:
    def __init__(self):
        # TODO: Update these transition probabilities with better values.
        self.transition_probabilities = {
            'insomnia': {
                1: [0.1, 0.2, 0.3, 0.2, 0.2],
                2: [0.2, 0.1, 0.2, 0.3, 0.2],
                3: [0.2, 0.2, 0.1, 0.2, 0.3],
                4: [0.3, 0.2, 0.2, 0.1, 0.2],
                5: [0.2, 0.3, 0.2, 0.2, 0.1]
            },
            'depression': {
                1: [0.1, 0.3, 0.2, 0.2, 0.2],
                2: [0.2, 0.1, 0.3, 0.2, 0.2],
                3: [0.2, 0.2, 0.1, 0.3, 0.2],
                4: [0.2, 0.2, 0.2, 0.1, 0.3],
                5: [0.3, 0.2, 0.2, 0.2, 0.1]
            },
            'anxiety': {
                1: [0.1, 0.2, 0.2, 0.3, 0.2],
                2: [0.2, 0.1, 0.3, 0.2, 0.2],
                3: [0.2, 0.2, 0.1, 0.2, 0.3],
                4: [0.3, 0.2, 0.2, 0.1, 0.2],
                5: [0.2, 0.3, 0.2, 0.2, 0.1]
            }
        }

    def update_state(self, current_state, state_type):
        # Use the transition probabilities to determine the next state based on the current state.
        return np.random.choice([1, 2, 3, 4, 5], p=self.transition_probabilities[state_type][current_state])

class Patient:
    def __init__(self, insomnia, depression, anxiety):
        self.insomnia = insomnia
        self.depression = depression
        self.anxiety = anxiety
        self.speech_fluency = self.calculate_speech_fluency()
        self.current_medication = None

    def calculate_speech_fluency(self):
        # This is a simple function that reduces speech fluency based on increased insomnia, depression, and anxiety.
        # This can be changed or refined as needed.
        return 1 - 0.1*(self.insomnia + self.depression + self.anxiety)

    def take_medication(self, medication):
        self.current_medication = medication

    def update_states(self):
        if self.current_medication:
            depression_effect, anxiety_effect, insomnia_effect, side_effect = self.current_medication.apply_effect()
            self.depression = max(1, min(5, self.depression + depression_effect))
            self.anxiety = max(1, min(5, self.anxiety + anxiety_effect))
            self.insomnia = max(1, min(5, self.insomnia + insomnia_effect))
            self.speech_fluency = max(0, min(1, self.speech_fluency - 0.05*side_effect))
            self.speech_fluency = self.calculate_speech_fluency()

# Incorporate HMM in the Patient class
class PatientWithHMM(Patient):
    def __init__(self, insomnia, depression, anxiety):
        super().__init__(insomnia, depression, anxiety)
        self.hmm = HiddenMarkovModel()

    def update_states(self):
        # Use HMM to update latent states
        self.insomnia = self.hmm.update_state(self.insomnia, 'insomnia')
        self.depression = self.hmm.update_state(self.depression, 'depression')
        self.anxiety = self.hmm.update_state(self.anxiety, 'anxiety')
        
        if self.current_medication:
            depression_effect, anxiety_effect, insomnia_effect, side_effect = self.current_medication.apply_effect()
            
            # Ensure the states are integers after updating with medication effects.
            self.depression = int(round(max(1, min(5, self.depression + depression_effect))))
            self.anxiety = int(round(max(1, min(5, self.anxiety + anxiety_effect))))
            self.insomnia = int(round(max(1, min(5, self.insomnia + insomnia_effect))))
            self.speech_fluency = max(0, min(1, self.speech_fluency - 0.05*side_effect))
        
        self.speech_fluency = self.calculate_speech_fluency()