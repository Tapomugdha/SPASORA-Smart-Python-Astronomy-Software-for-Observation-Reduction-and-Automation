# dithering.py
import numpy as np

class DitherManager:
    def __init__(self, max_dither_pixels=5):
        self.max_dither_pixels = max_dither_pixels
        self.dither_offset = (0, 0)

    def random_dither(self):
        dx = np.random.uniform(-self.max_dither_pixels, self.max_dither_pixels)
        dy = np.random.uniform(-self.max_dither_pixels, self.max_dither_pixels)
        self.dither_offset = (dx, dy)
        return self.dither_offset

    def apply_dither(self, reference_pos):
        # Returns a new reference position offset by the current dither
        return (reference_pos[0] + self.dither_offset[0], reference_pos[1] + self.dither_offset[1])

    def reset(self):
        self.dither_offset = (0, 0)
