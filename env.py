import numpy as np

class Medication:
    def __init__(self, name, depression_effect, anxiety_effect, insomnia_effect, side_effect, time_to_effect):
        self.name = name
        self.depression_effect = depression_effect
        self.anxiety_effect = anxiety_effect
        self.insomnia_effect = insomnia_effect
        self.side_effect = side_effect
        self.time_to_effect = time_to_effect

    def __repr__(self) -> str:
        return self.name

class SimulationEnvironment:
    def __init__(self, patient, medications):
        self.patient = patient
        self.medications = medications
        self.day = 0

    def _apply_medication_effects(self, medication):
        mean, std = medication.depression_effect
        self.patient.depression += np.random.normal(mean, std)
        
        mean, std = medication.anxiety_effect
        self.patient.anxiety += np.random.normal(mean, std)
        
        mean, std = medication.insomnia_effect
        self.patient.insomnia += np.random.normal(mean, std)
        
        mean, std = medication.side_effect
        self.patient.side_effect_accumulation += np.random.normal(mean, std)
        
        self.patient.medication_accumulation += 1
        
    def step(self, action):
        chosen_medication = self.medications[action]
        self._apply_medication_effects(chosen_medication)
        
        # Update patient's speech fluency based on the current state
        self.patient.speech_fluency = 1 - (0.2 * self.patient.depression + 0.3 * self.patient.anxiety + 0.1 * self.patient.insomnia)/15
        
        # Calculate reward as the speech fluency minus side effects
        reward = self.patient.speech_fluency - (0.2 * self.patient.side_effect_accumulation)
        
        context = np.array([self.patient.speech_fluency, self.patient.hours_slept_previous_night, self.patient.medication_accumulation, self.day])
        self.day += 1
        
        return context, reward