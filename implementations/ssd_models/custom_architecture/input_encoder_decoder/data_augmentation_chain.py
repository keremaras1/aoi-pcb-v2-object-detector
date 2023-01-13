import numpy as np
from PIL import Image
import random
import cv2
from tqdm import tqdm


class DataAugmentationChain:
    def __init__(self,
                 image_array,
                 unencoded_label_array,
                 probability):
        self.X = image_array
        self.y = unencoded_label_array
        self.probability = probability

    def __call__(self, *args, **kwargs):
        print('Applying randomized augmentation...')
        for i in tqdm(range(len(self.X))):
            self.X[i], self.y[i] = self.vertical_flip(self.X[i], self.y[i])
            self.X[i], self.y[i] = self.horizontal_flip(self.X[i], self.y[i])
            self.X[i], self.y[i] = self.perpendicular_rotate(self.X[i], self.y[i])
            self.X[i], self.y[i] = self.random_brightness(self.X[i], self.y[i])
            self.X[i], self.y[i] = self.random_contrast(self.X[i], self.y[i])
            self.X[i], self.y[i] = self.random_hue(self.X[i], self.y[i])
            self.X[i], self.y[i] = self.random_lighting_noise(self.X[i], self.y[i])

        return self.X, self.y

    def horizontal_flip(self, image, label):
        decision = random.random() < self.probability

        if not decision:
            return image, label

        img = np.fliplr(image)

        label[:, [-10, -8, -6, -4, -2]] = image.shape[1] - label[:, [-10, -8, -6, -4, -2]]

        return img, label

    def vertical_flip(self, image, label):
        decision = random.random() < self.probability

        if not decision:
            return image, label

        img = np.flipud(image)

        label[:, [-9, -7, -5, -3, -1]] = image.shape[0] - label[:, [-9, -7, -5, -3, -1]]

        return img, label

    def perpendicular_rotate(self, image, label):
        decision = random.random() < self.probability

        if not decision:
            return image, label

        angel_list = [90, 180, 270]
        angle = random.choice(angel_list)

        img = Image.fromarray(image)
        rotated_img = img.rotate(angle)
        rotated_img = np.array(rotated_img)

        rotated_height, rotated_width, _ = rotated_img.shape

        for i in range(label.shape[0]):
            coords_2d = np.reshape(label[i, -10:], (-1, 2))
            if angle == 90:
                rotated_coords = np.array([coords_2d[:, 1], rotated_width - coords_2d[:, 0]]).T
            elif angle == 180:
                rotated_coords = np.array([rotated_height - coords_2d[:, 0], rotated_width - coords_2d[:, 1]]).T
            elif angle == 270:
                rotated_coords = np.array([rotated_width - coords_2d[:, 1], coords_2d[:, 0]]).T

            coords = np.reshape(rotated_coords, label[i, -10:].shape)
            label[i, -10:] = coords

        return rotated_img, label

    def random_brightness(self, image, label, min_delta=-75, max_delta=75):
        decision = random.random() < self.probability

        if not decision:
            return image, label

        img = image.copy().astype(float)
        d = random.uniform(min_delta, max_delta)
        img += d
        img = np.clip(img, 0, 255)

        return np.uint8(img), label

    def random_contrast(self, image, label, min_delta=0.5, max_delta=1.8):
        decision = random.random() < self.probability

        if not decision:
            return image, label

        img = np.copy(image).astype(float)
        d = random.uniform(min_delta, max_delta)
        img *= d
        img = np.clip(img, 0, 255)

        return np.uint8(img), label

    def random_hue(self, image, label, min_delta=-18, max_delta=18):
        decision = random.random() < self.probability

        if not decision:
            return image, label

        img = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2HSV)
        img = np.array(img).astype(float)
        d = random.uniform(min_delta, max_delta)
        img[:, :, 0] += d
        img = np.clip(img, 0, 360)
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_HSV2RGB)
        img = np.array(img)

        return np.uint8(img), label

    def random_lighting_noise(self, image, label):
        decision = random.random() < self.probability

        if not decision:
            return image, label

        img = np.copy(image)

        perms = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0)
        ]

        selected_perm = random.randint(0, len(perms) - 1)
        perm = perms[selected_perm]
        img = img[:, :, perm]

        return img, label
