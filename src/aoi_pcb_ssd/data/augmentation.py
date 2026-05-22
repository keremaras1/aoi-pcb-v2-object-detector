"""Data augmentation chain for synthetic PCB training images.

Each image is assigned exactly one augmentation drawn uniformly from six
photometric and geometric transforms. Whether that transform is actually
applied is then gated by ``probability``, so the effective augmentation rate
is ``1/6 * probability`` per transform type.
"""

import random

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


class DataAugmentationChain:
    """Apply one randomly selected augmentation to each image in a dataset.

    Six transforms are available: brightness, contrast, lighting noise,
    vertical flip, horizontal flip, and perpendicular rotation (90°/180°/270°).
    Each image receives at most one transform, chosen uniformly at random and
    then gated by ``probability``.

    The label format expected is a list of arrays of shape ``(n_ics, 11)``,
    where the last 10 columns are x/y coordinate pairs in the order
    ``tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, cx, cy``.

    Args:
        image_array: Uint8 image array of shape ``(N, H, W, C)``.
        unencoded_label_array: List of ``N`` label arrays, one per image.
        probability: Gate probability for each selected transform.
            A value of 0.1 means the chosen transform is applied to 10% of
            images even after selection.
        seed: Random seed for reproducibility.
    """

    _AUGMENTATIONS = [
        "random_brightness",
        "random_contrast",
        "random_lighting_noise",
        "vertical_flip",
        "horizontal_flip",
        "perpendicular_rotate",
    ]

    def __init__(
        self,
        image_array: NDArray[np.uint8],
        unencoded_label_array: list,
        probability: float,
        seed: int,
    ) -> None:
        self.X = image_array
        self.y = unencoded_label_array
        self.probability = probability
        self.seed = seed

    def __call__(self) -> tuple[NDArray[np.uint8], list]:
        """Apply augmentation to all images and return the modified arrays.

        Returns:
            Tuple of ``(image_array, label_array)`` after augmentation.
        """
        random.seed(self.seed)
        augment_fns = [getattr(self, name) for name in self._AUGMENTATIONS]

        for i in tqdm(range(len(self.X)), desc="Augmenting"):
            fn = random.choice(augment_fns)
            self.X[i], self.y[i] = fn(self.X[i], self.y[i])

        return self.X, self.y

    def horizontal_flip(
        self, image: NDArray[np.uint8], label: NDArray
    ) -> tuple[NDArray[np.uint8], NDArray]:
        """Flip image left-right and mirror all x-coordinates."""
        if random.random() >= self.probability:
            return image, label
        img = np.fliplr(image)
        label[:, [-10, -8, -6, -4, -2]] = image.shape[1] - label[:, [-10, -8, -6, -4, -2]]
        return img, label

    def vertical_flip(
        self, image: NDArray[np.uint8], label: NDArray
    ) -> tuple[NDArray[np.uint8], NDArray]:
        """Flip image top-bottom and mirror all y-coordinates."""
        if random.random() >= self.probability:
            return image, label
        img = np.flipud(image)
        label[:, [-9, -7, -5, -3, -1]] = image.shape[0] - label[:, [-9, -7, -5, -3, -1]]
        return img, label

    def perpendicular_rotate(
        self, image: NDArray[np.uint8], label: NDArray
    ) -> tuple[NDArray[np.uint8], NDArray]:
        """Rotate image by 90°, 180°, or 270° and update corner coordinates.

        Uses ``np.rot90`` for lossless rotation (no interpolation artefacts).
        """
        if random.random() >= self.probability:
            return image, label

        angle = random.choice([90, 180, 270])
        k = angle // 90
        image = np.rot90(image, k=k, axes=(0, 1))
        h, w, _ = image.shape

        for i in range(label.shape[0]):
            coords = np.reshape(label[i, -10:], (-1, 2))
            if angle == 90:
                rotated = np.array([coords[:, 1], w - coords[:, 0]]).T
            elif angle == 180:
                rotated = np.array([h - coords[:, 0], w - coords[:, 1]]).T
            else:  # 270
                rotated = np.array([w - coords[:, 1], coords[:, 0]]).T
            label[i, -10:] = np.reshape(rotated, label[i, -10:].shape)

        return image, label

    def random_brightness(
        self,
        image: NDArray[np.uint8],
        label: NDArray,
        min_delta: float = -50.0,
        max_delta: float = 50.0,
    ) -> tuple[NDArray[np.uint8], NDArray]:
        """Add a random brightness offset to all pixels."""
        if random.random() >= self.probability:
            return image, label
        d = random.uniform(min_delta, max_delta)
        image = np.clip(image.astype(float) + d, 0, 255).astype(np.uint8)
        return image, label

    def random_contrast(
        self,
        image: NDArray[np.uint8],
        label: NDArray,
        min_delta: float = 0.5,
        max_delta: float = 1.5,
    ) -> tuple[NDArray[np.uint8], NDArray]:
        """Multiply all pixel values by a random contrast factor."""
        if random.random() >= self.probability:
            return image, label
        d = random.uniform(min_delta, max_delta)
        image = np.clip(image.astype(float) * d, 0, 255).astype(np.uint8)
        return image, label

    def random_lighting_noise(
        self, image: NDArray[np.uint8], label: NDArray
    ) -> tuple[NDArray[np.uint8], NDArray]:
        """Randomly permute the RGB colour channels."""
        if random.random() >= self.probability:
            return image, label
        perms = [
            (0, 1, 2), (0, 2, 1), (1, 0, 2),
            (1, 2, 0), (2, 0, 1), (2, 1, 0),
        ]
        perm = perms[random.randint(0, len(perms) - 1)]
        return np.copy(image[:, :, perm]), label
