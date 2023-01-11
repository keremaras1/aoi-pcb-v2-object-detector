import numpy as np
from PIL import Image
import random


class DataAugmentationChain:
    def __init__(self,
                 image_array,
                 unencoded_label_array,
                 probability):
        self.X = image_array
        self.y = unencoded_label_array
        self.probability = probability

    def horizontal_flip(self, image, label):
        decision = random.random() < self.probability

        if not decision:
            return image, label

        img = Image.fromarray(image)

        
