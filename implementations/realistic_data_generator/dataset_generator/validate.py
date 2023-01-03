import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image

if __name__ == '__main__':
    dir_name = '/Users/keremaras/Desktop/test/cropped_dataset'
    dir_list = os.listdir(dir_name)
    labels = pd.read_csv('/Users/keremaras/Desktop/test/cropped_dataset/labels.csv')

    for i in range(10):
        if dir_list[i].endswith('.jpg'):
            img = Image.open(os.path.join(dir_name, dir_list[i]))

            label_df = labels.loc[labels['frame'] == dir_list[i]]
            coords = label_df.drop(columns=['frame', 'class_id']).to_numpy()
            coords = np.reshape(coords, (-1, 2))

            for coord in coords:
                img = cv2.circle(np.ascontiguousarray(img), tuple(coord), 4, (0, 0, 255), 2)

            Image.fromarray(img).show()
