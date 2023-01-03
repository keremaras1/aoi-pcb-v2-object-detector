import math
import re
import os
import argparse
import random
import cv2
import numpy as np
import pandas as pd
from template import Template
from tqdm import tqdm
from PIL import Image


def image_resize(img, save_path, size):
    img1 = img.resize(size)
    img1.save(save_path)


def centroids_to_corners(centroids):
    x_min = centroids[:, 0] - centroids[:, 2] / 2
    x_max = centroids[:, 0] + centroids[:, 2] / 2
    y_min = centroids[:, 1] - centroids[:, 3] / 2
    y_max = centroids[:, 1] + centroids[:, 3] / 2

    corners = np.array([x_min, y_min, x_max, y_min, x_min, y_max, x_max, y_max, centroids[:, 0], centroids[:, 1]]).T
    return np.rint(corners).astype(int)


def rotate_coords(coords, angle):
    center = coords[-1]
    for i, corner in enumerate(coords[:-1]):
        coords[i] = rotate(center, corner, math.radians(-angle))
    return np.reshape(coords, (-1,))


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """

    ox, oy = origin[0], origin[1]
    px, py = point[0], point[1]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def get_new_corners(corners, offset_x, offset_y, rotation):
    corners_mat = np.reshape(corners, (-1, 2))
    shift_arr = np.array([offset_x, offset_y])

    corners_mat += shift_arr

    new_corners = rotate_coords(corners_mat, rotation)
    return new_corners


def check_ic_in_crop(bounds, coords):
    idx = []

    x_coords = coords[:, [0, 2, 4, 6, 8]]
    y_coords = coords[:, [1, 3, 5, 7, 9]]

    for i in range(coords.shape[0]):
        x_bounds = np.logical_and(x_coords[i] >= bounds[0], x_coords[i] < bounds[2])
        y_bounds = np.logical_and(y_coords[i] >= bounds[1], y_coords[i] < bounds[3])

        if np.logical_and(np.all(x_bounds), np.all(y_bounds)):
            idx.append(i)

    return idx


def get_new_coords_for_crop(crop_xmin, crop_ymin, corner):
    corner[:, [0, 2, 4, 6, 8]] -= crop_xmin
    corner[:, [1, 3, 5, 7, 9]] -= crop_ymin

    return corner


class pcb_dataset_generator:
    def __init__(self,
                 template_dir,
                 save_dir,
                 crop_save_dir,
                 rotation_range,
                 img_size,
                 dataset_size,
                 placement_offset_x,
                 placement_offset_y):
        self.template_main_dir = template_dir
        self.save_dir = save_dir
        self.crop_save_dir = crop_save_dir
        self.template_object_list = []
        self.rotation_range = rotation_range
        self.img_size = img_size
        self.dataset_size = dataset_size
        self.offsets = [placement_offset_x, placement_offset_y]
        self.create_template_objects()
        self.labels = pd.DataFrame(
            columns=['frame', 'tl_x', 'tl,y', 'tr_x', 'tr_y', 'bl_x', 'bl_y', 'br_x', 'br_y', 'cx', 'cy', 'class_id'])
        self.crop_labels = pd.DataFrame(
            columns=['frame', 'tl_x', 'tl,y', 'tr_x', 'tr_y', 'bl_x', 'bl_y', 'br_x', 'br_y', 'cx', 'cy', 'class_id'])

    def create_template_objects(self):
        for template_dir in os.listdir(self.template_main_dir):
            if template_dir == '.DS_Store':
                continue
            for p in os.listdir(os.path.join(self.template_main_dir, template_dir)):
                if (re.search(r"\.DS_Store", p) or re.search(r".*\.csv$", p) or re.search(r".*\.jpg$", p)) is not None:
                    continue
                t_object = Template(os.path.join(self.template_main_dir, template_dir, p))
                self.template_object_list.append(t_object)

    def get_rotation(self):
        return np.random.normal(0, self.rotation_range)

    def get_ic_placement(self, cx, cy, rotated_ic_cutout):
        width, height = rotated_ic_cutout.size
        offset_x = np.random.randint(-self.offsets[0], self.offsets[0])
        offset_y = np.random.randint(-self.offsets[1], self.offsets[1])

        placement_x = int(cx - width / 2)
        placement_y = int(cy - height / 2)

        return [placement_x + offset_x, placement_y + offset_y, offset_x, offset_y]

    def generate_image_and_corners(self, image_name):
        template = random.choice(self.template_object_list)
        background = template.get_background().copy()
        ic_cutouts = template.get_ic_cutouts()
        labels_df = template.get_labels_df()

        centroids = labels_df.loc[:, ['cx', 'cy', 'w', 'h']].to_numpy()
        corners = centroids_to_corners(centroids)

        for i, ic in enumerate(ic_cutouts):
            rotation = self.get_rotation()
            im1 = ic.convert('RGBA')
            rotated_im1 = im1.rotate(rotation, expand=True)
            placement = self.get_ic_placement(centroids[i, 0], centroids[i, 1], rotated_im1)
            corners[i] = get_new_corners(corners[i], placement[2], placement[3], rotation)
            background.paste(rotated_im1, (int(placement[0]), int(placement[1])), rotated_im1)

        background.save(os.path.join(self.save_dir, image_name))
        img_df = pd.DataFrame(corners,
                              columns=['tl_x', 'tl,y', 'tr_x', 'tr_y', 'bl_x', 'bl_y', 'br_x', 'br_y', 'cx', 'cy'])

        class_id = np.ones(corners.shape[0], dtype=int).tolist()
        frame = np.full(corners.shape[0], image_name).tolist()

        img_df.insert(0, 'frame', frame, True)
        img_df.insert(len(img_df), 'class_id', class_id, True)

        self.labels = pd.concat([self.labels, img_df])

    def save_labels(self, version='uncropped'):
        assert version == 'uncropped' or version == 'cropped'

        if version == 'uncropped':
            self.labels.to_csv(os.path.join(self.save_dir, 'labels.csv'), index=False)
        else:
            self.crop_labels.to_csv(os.path.join(self.crop_save_dir, 'labels.csv'), index=False)

    def generate(self, version='uncropped'):
        assert version == 'uncropped' or version == 'cropped'

        if version == 'uncropped':
            print('Generating images and labels: ')
            for i in tqdm(range(self.dataset_size)):
                label = 'PCB_{idx}.jpg'.format(idx=i)
                self.generate_image_and_corners(label)
            self.save_labels()
        else:
            img_list = os.listdir(self.save_dir)
            i = 0
            pbar = tqdm(desc='Generating cropped images and labels: ', total=self.dataset_size)
            while i < self.dataset_size:
                img = random.choice(img_list)
                if not img.endswith('.jpg'):
                    continue
                label = 'PCB_crop_{idx}.jpg'.format(idx=i)
                success = self.generate_crop(img_path=os.path.join(self.save_dir, img), crop_name=label)
                if not success:
                    continue
                i += 1
                pbar.update(1)
            self.save_labels(version='cropped')

    def get_coords_from_df(self, img_name):
        label_df = self.labels.loc[self.labels['frame'] == img_name]
        coords = label_df.drop(columns=['frame', 'class_id']).to_numpy()

        return coords

    def generate_crop(self, img_path, crop_name):
        img = Image.open(img_path)
        width, height = img.size

        xmin = np.random.randint(0, width)
        ymin = np.random.randint(0, height)
        xmax = min(width, int(xmin + 0.4 * width))
        ymax = min(height, int(ymin + 0.4 * width))

        while (xmax - xmin) / (ymax - ymin) != 1:
            xmin = np.random.randint(0, width)
            ymin = np.random.randint(0, height)
            xmax = min(width, int(xmin + 0.7 * width))
            ymax = min(height, int(ymin + 0.7 * width))

        bounds = np.array([xmin, ymin, xmax, ymax])
        parent_dir, img_name = os.path.split(img_path)

        corners = self.get_coords_from_df(img_name)

        ic_idx = check_ic_in_crop(bounds, corners)

        if len(ic_idx) <= 0:
            return False

        crop_img = img.crop((xmin, ymin, xmax, ymax))
        crop_width, crop_height = crop_img.size

        valid_corners = corners[ic_idx, :]
        new_valid_corners = get_new_coords_for_crop(xmin, ymin, valid_corners)

        if self.img_size not in [crop_width, crop_height]:
            crop_img = crop_img.resize((self.img_size, self.img_size))
            new_valid_corners = self.resize_coords(crop_width, crop_height, new_valid_corners)

        crop_label_df = pd.DataFrame(new_valid_corners,
                                     columns=['tl_x', 'tl,y', 'tr_x', 'tr_y', 'bl_x', 'bl_y', 'br_x', 'br_y', 'cx',
                                              'cy'])

        class_id = np.ones(new_valid_corners.shape[0], dtype=int).tolist()
        frame = np.full(new_valid_corners.shape[0], crop_name).tolist()

        crop_label_df.insert(0, 'frame', frame, True)
        crop_label_df.insert(len(crop_label_df), 'class_id', class_id, True)

        self.crop_labels = pd.concat([self.crop_labels, crop_label_df])

        crop_img.save(os.path.join(self.crop_save_dir, crop_name))

        return True

    def get_csv_entry_for_img(self):
        # TODO: generify label creation (optional)
        return

    def resize_coords(self, org_width, org_height, coords):
        w_ratio = self.img_size / org_width
        h_ratio = self.img_size / org_height

        coords[:, [0, 2, 4, 6, 8]] *= w_ratio
        coords[:, [1, 3, 5, 7, 9]] *= h_ratio

        return coords.astype(int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('template_dir')
    parser.add_argument('save_dir')
    parser.add_argument('crop_save_dir')
    parser.add_argument('rotation_range', type=int)
    parser.add_argument('img_size', type=int)
    parser.add_argument('dataset_size', type=int)
    parser.add_argument('placement_offset_x', type=int)
    parser.add_argument('placement_offset_y', type=int)
    args = parser.parse_args()

    generator = pcb_dataset_generator(args.template_dir, args.save_dir, args.crop_save_dir, args.rotation_range,
                                      args.img_size,
                                      args.dataset_size, args.placement_offset_x, args.placement_offset_y)
    generator.generate()
    generator.generate(version='cropped')
