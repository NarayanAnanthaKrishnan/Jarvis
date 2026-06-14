import sounddevice as sd
import numpy as np


class Player:
    def __init__(self):
        pass

    def play(self, samples: np.ndarray, sample_rate: int = 16000):
        sd.play(samples, sample_rate)
        sd.wait()
