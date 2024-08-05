# Layers/Base.py

class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.testing_phase = False

    @property
    def phase(self):
        return self.testing_phase

    @phase.setter
    def phase(self, phase):
        self.testing_phase = phase

    