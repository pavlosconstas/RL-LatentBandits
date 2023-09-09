import numpy as np
from hmmlearn import hmm

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

        return current_depression_obs, current_anxiety_obs, current_insomnia_obs
    
    def administer_medication(self, medication, current_day):

        medication.day_administered = current_day
        self.medication_accumulation.append(medication)

    def update_medication_dosage(self, current_day):
        for medication in self.medication_accumulation:
            medication.update_dosage(current_day)
        
        self.medication_accumulation = [med for med in self.medication_accumulation if med.current_dosage > 0.005]

    def apply_medication_effects(self, current_day):

        pass



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
    
    def is_active(self, current_day):
        self.is_active = current_day >= self.day_administered + self.time_to_effect
    
    def update_dosage(self, current_day):
        days_since_administered = current_day - self.day_administered

        if days_since_administered % self.half_life == 0:
            self.dosage /= 2

    def __str__(self) -> str:
        return f"Name: {self.name}\nDepression Effect: {self.depression_effect}\nAnxiety Effect: {self.anxiety_effect}\nInsomnia Effect: {self.insomnia_effect}\nSide Effect: {self.side_effect}\nTime to Effect: {self.time_to_effect}\nHalf Life: {self.half_life}\n"

class SimulationEnvironment:

    def __init__(self, patient, medication_list) -> None:
        
        self.patient = patient
        self.medication_list = medication_list

    def step(self, action):
        pass

patient = Patient()
print(patient)
print(patient.depression_model.transmat_)
print(patient.anxiety_model.transmat_)
print(patient.insomnia_model.transmat_)


for _ in range(100):
    depression_X, anxiety_X, insomnia_X = patient.generate_one_step()
    print(f'Current Depression State: {patient.current_depression_state} Current Anxiety State: {patient.current_anxiety_state} Current Insomnia State: {patient.current_insomnia_state}')
    print(f'Depression Observation: {depression_X} Anxiety Observation: {anxiety_X} Insomnia Observation: {insomnia_X}')
    print(f'Current Speech Fluency: {patient.current_speech_fluency}')
