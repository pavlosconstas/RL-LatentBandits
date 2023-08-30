import numpy as np

class Patient:
    def __init__(self):
        self.depression = np.random.randint(1, 6)  # Randomly initialized between 1-5
        self.anxiety = np.random.randint(1, 6)     # Randomly initialized between 1-5
        self.insomnia = np.random.randint(1, 6)    # Randomly initialized between 1-5
        self.speech_fluency = np.random.random()   # Randomly initialized between 0-1
        self.medication_accumulation = 0
        self.hours_slept_previous_night = max(0, np.random.normal(8 - self.insomnia, 2))  # Influenced by insomnia score
        self.side_effect_accumulation = 0