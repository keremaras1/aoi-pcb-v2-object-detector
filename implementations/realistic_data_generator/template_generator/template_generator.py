import argparse
import os
import cv2
import pandas as pd
import numpy as np


def read_corners_from_csv(coords_path):
    print('Reading Coordinates...')
    df = pd.read_csv(coords_path)
    coords = df.to_numpy()
    return coords


def convert_coords(corner_coords):
    xmin = corner_coords[:, 0]
    ymin = corner_coords[:, 1]
    xmax = corner_coords[:, 2]
    ymax = corner_coords[:, 3]

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    centroids = np.array([cx, cy, w, h]).T

    return centroids


def get_dominant_color(img):
    height, width, _ = np.shape(img)

    data = np.reshape(img, (height * width, 3))
    data = np.float32(data)

    n_clusters = 1
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, n_clusters, None, crit, 10, flags)
    return centers[0]


class template_generator:
    def __init__(self, pcb_img_path, ic_coords_path, template_object_path):
        self.pcb_img = cv2.imread(pcb_img_path)
        self.ic_coords_corners = read_corners_from_csv(ic_coords_path)
        self.ic_coords_centroids = convert_coords(self.ic_coords_corners)
        self.save_dir = template_object_path
        self.labels = []
        self.color = get_dominant_color(self.pcb_img)

    def generate_ic_cutouts(self):
        print('Generating IC Cutouts...')
        for i, corners in enumerate(self.ic_coords_corners):
            cropped_ic = self.pcb_img[corners[1]:corners[3], corners[0]:corners[2], :]
            name = 'cropped_ic_{idx}.jpg'.format(idx=i)
            cv2.imwrite(os.path.join(self.save_dir, name), cropped_ic)
            label = [name, *map(float, self.ic_coords_centroids[i])]
            self.labels.append(label)

    def fill_cutouts(self):
        background = self.pcb_img.copy()
        for corners in self.ic_coords_corners:
            background[corners[1]:corners[3], corners[0]:corners[2], :] = self.color

        cv2.imwrite(os.path.join(self.save_dir, 'background_pcb.jpg'), background)

    def save_labels(self):
        print('Saving new IC Coordinates...')
        self.labels = np.array(self.labels)
        df = pd.DataFrame(self.labels)
        header = ['ic_frame', 'cx', 'cy', 'w', 'h']
        df.to_csv(os.path.join(self.save_dir, 'ic_centroids.csv'), header=header, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('parent_dir')
    parser.add_argument('img_name')
    parser.add_argument('ic_coords')
    parser.add_argument('template_object_path')
    args = parser.parse_args()

    pcb_path = os.path.join(args.parent_dir, args.img_name)
    ic_path = os.path.join(args.parent_dir, args.ic_coords)

    generator = template_generator(pcb_path, ic_path, args.template_object_path)
    generator.generate_ic_cutouts()
    generator.save_labels()
    generator.fill_cutouts()
