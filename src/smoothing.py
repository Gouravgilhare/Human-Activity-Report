
from collections import deque

import numpy as np


class TemporalSmoother:
    def __init__(self, window_size=9):
        self.window = deque(maxlen=window_size)

    def update(self, probabilities, classes):
        probabilities = np.asarray(probabilities, dtype=float)

        if not np.isfinite(probabilities).all():
            return "Uncertain", 0.0

        self.window.append(probabilities)

        # Average predictions over recent frames.
        averaged = np.mean(np.stack(self.window), axis=0)
        best_index = int(np.argmax(averaged))
        confidence = float(averaged[best_index])

        # Avoid displaying a definite label for weak predictions.
        if confidence < 0.55:
            return "Uncertain", confidence

        return str(classes[best_index]), confidence

    def reset(self):
        self.window.clear()
