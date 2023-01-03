import re
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd


class Template:
    def __init__(self,
                 template_path):
        self.template_path = template_path
        self.label_csv = None
        self.ic_cutouts = []
        self.background = None
        self.sort_alphanumeric()

    def sort_alphanumeric(self):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        sorted_dir = sorted(os.listdir(self.template_path), key=alphanum_key)

        for name in sorted_dir:
            if re.search(r".*\.jpg$", name) is not None:
                if name.startswith("background"):
                    self.background = Image.open(os.path.join(self.template_path, name))
                else:
                    ic = Image.open(os.path.join(self.template_path, name))
                    self.ic_cutouts.append(ic)
            elif re.search(r".*\.csv$", name) is not None:
                self.label_csv = pd.read_csv(os.path.join(self.template_path, name))

    def get_template_path(self):
        return self.template_path

    def get_labels_df(self):
        return self.label_csv

    def get_ic_cutouts(self):
        return self.ic_cutouts

    def get_background(self):
        return self.background
