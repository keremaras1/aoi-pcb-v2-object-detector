import numpy as np
import pandas as pd
from PIL import Image
import re
import os
from tqdm import tqdm
from PIL import Image, ImageDraw

from input_encoder_decoder.input_encoder_new import SSDInputEncoder
from input_encoder_decoder.data_augmentation_chain import DataAugmentationChain


class DataGenerator:
    def __init__(self, parent_dir, encoder, augmentation=True, probability=0.5):
        self.parent_dir = parent_dir
        self.encoder = encoder
        self.img_filenames = []
        self.sort_alphanumeric()
        self.X = []
        self.y = []
        self.y_encoded = []
        self.augmentation = augmentation
        self.probability = probability

    def sort_alphanumeric(self):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        sorted_dir = sorted(os.listdir(self.parent_dir), key=alphanum_key)

        for name in sorted_dir:
            if re.search(r".*\.jpg$", name) is not None:
                self.img_filenames.append(name)

    def img_to_np(self):
        print('Converting images to arrays...')
        for img in tqdm(self.img_filenames):
            pcb = Image.open(os.path.join(self.parent_dir, img))
            x = np.array(pcb)
            self.X.append(x)
        self.X = np.array(self.X)
        print('Images as numpy:')
        print(self.X.shape)

    def parse_csv(self, csv_name):
        df_main = pd.read_csv(os.path.join(self.parent_dir, csv_name))
        img_names_unique = df_main['frame'].unique()

        print('Parsing ground truth labels from .csv')
        for name in tqdm(img_names_unique):
            labels = df_main[df_main['frame'] == name]
            gt_corners = labels.iloc[:, [11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
            gt_unencoded = gt_corners.to_numpy()
            self.y.append(gt_unencoded)
        print('Unencoded labels:')
        print(len(self.y))

    def generate_y(self):
        self.parse_csv('labels.csv')
        if self.augmentation:
            self.augment_data()
        self.y_encoded = self.encoder(self.y)
        print('Encoded labels:')
        print(self.y_encoded.shape)

    def augment_data(self):
        print('Augmenting images and relabeling...')
        augmentator = DataAugmentationChain(self.X, self.y, probability=self.probability, seed=42)
        self.X, self.y = augmentator()

    def get_data(self):
        print('Generating image arrays and encoding labels...')
        self.img_to_np()
        self.generate_y()
        return self.X, self.y_encoded


if __name__ == "__main__":
    print("Data generator is running as __main__!!!")
    encoder = SSDInputEncoder(img_height=300,
                              img_width=300,
                              n_classes=1,
                              predictor_sizes=[(10, 10)],
                              normalize_coords=True,
                              background_id=0)

    generator = DataGenerator(parent_dir='/home/kerem/AOI_Project/Datasets/real_pcb_crop', encoder=encoder)

    X, y = generator.get_data()
    print(X.shape)
    print(y.shape)
