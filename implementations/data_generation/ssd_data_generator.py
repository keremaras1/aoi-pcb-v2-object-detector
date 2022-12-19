import os
import cv2
import math
import argparse
import numpy as np
import pandas as pd
import random
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('rotation_range', type=int)
parser.add_argument('ic_count', type=int)
parser.add_argument('ratio', type=float)
parser.add_argument('size', type=int)
parser.add_argument('dataset_size', type=int)
parser.add_argument('parent_dir')
parser.add_argument('version')
args = parser.parse_args()


def image_resize(img_path, save_path, size):
    img = Image.open(img_path)
    img1 = img.resize(size)
    img1.save(save_path)


def generate_green_background(img_path, generate=False):
    if generate:
        img = Image.open(img_path)
        green = Image.new('RGBA', img.size, color='green')
        green.save(os.path.join(os.path.dirname(img_path), 'green_background.png'))
        print("Background generated")


def center_to_corners(ic_center, w, h, version):
    x_min = ic_center[0] - w / 2
    x_max = ic_center[0] + w / 2
    y_min = ic_center[1] - h / 2
    y_max = ic_center[1] + h / 2
    if version == '1':
        return [x_min, y_min, x_max, y_max]
    else:
        return [x_min, y_min, x_max, y_min, x_min, y_max, x_max, y_max, ic_center[0], ic_center[1]]


def rotate_coords(coords, angle):
    corners_mat = np.array(coords).reshape((-1, 2))
    center = corners_mat[-1]
    for i, corner in enumerate(corners_mat[:-1]):
        corners_mat[i] = rotate(center, corner, math.radians(-angle))
    return np.reshape(corners_mat, (-1,))


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


def shift_gt_coords(coords, shift):
    coords = np.asarray(coords)
    shift_arr = np.asarray(shift)

    dx = shift_arr[0]
    dy = shift_arr[1]

    x_min = coords[0] + dx
    y_min = coords[1] + dy
    x_max = coords[2] + dx
    y_max = coords[3] + dy
    return [x_min, y_min, x_max, y_max]


def get_gt_box(ic):
    img_arr = np.asarray(ic)
    h, w = img_arr.shape[0], img_arr.shape[1]
    x_min = 0
    y_min = 0
    x_max = x_min + w
    y_max = y_min + h
    return [x_min, y_min, x_max, y_max]


def shift_ic_corners(corners, shift):
    corner_shape = np.asarray(corners).shape
    corners_mat = np.reshape(corners, (-1, 2))
    shift_arr = np.asarray(shift)

    corners_mat += shift_arr

    corners_mat = np.reshape(corners_mat, corner_shape)
    return corners_mat


def is_overlapping(coords1, coords2):
    if coords1[0] > coords2[2] or coords2[0] > coords1[2]:
        return False

    if coords1[1] > coords2[3] or coords2[1] > coords1[3]:
        return False

    return True


def split_labels(labels_parentdir, split_ratio=0.7):
    print('Creating training and validation split...')
    assert 1.0 > split_ratio > 0, 'Ratio should be between 0 and 1 but instead the given ratio is {r}'.format(
        r=split_ratio)

    print('Split Ratio:', split_ratio)

    all_labels = pd.read_csv(os.path.join(labels_parentdir, "labels.csv"))
    n_labels = all_labels.shape[0]

    df_train = all_labels[:int(n_labels * split_ratio)]
    df_val = all_labels[int(n_labels * split_ratio):]

    df_train.to_csv(os.path.join(labels_parentdir, 'labels_train.csv'), index=False)
    df_val.to_csv(os.path.join(labels_parentdir, 'labels_val.csv'), index=False)


def edit_classes(labels_parentdir):
    train_df = pd.read_csv(os.path.join(labels_parentdir, "labels_train.csv"))
    val_df = pd.read_csv(os.path.join(labels_parentdir, "labels_trainval.csv"))
    trainval_df = pd.read_csv(os.path.join(labels_parentdir, "labels_val.csv"))

    train_df['class_id'] = train_df['class_id'].replace([0], 1)
    trainval_df['class_id'] = trainval_df['class_id'].replace([0], 1)
    val_df['class_id'] = val_df['class_id'].replace([0], 1)

    train_df.to_csv(os.path.join(labels_parentdir, 'labels_train.csv'), index=False)
    trainval_df.to_csv(os.path.join(labels_parentdir, 'labels_trainval.csv'), index=False)
    val_df.to_csv(os.path.join(labels_parentdir, 'labels_val.csv'), index=False)


def get_rotation_center(rotated_ic):
    w, h = rotated_ic.size
    cx = w / 2
    cy = h / 2
    return np.array((cx, cy))


class SSD_Data_Generator:
    def __init__(self, ic, background, rotation_range, ic_count, black=True, size=(300, 300)):
        self.background_ref = Image.open(background).resize(size)
        self.background = self.background_ref.copy()
        if black:
            self.ic = Image.new('RGBA', size=size, color='black')
        else:
            self.ic = Image.open(ic).resize(size)
        self.ic_count = ic_count
        self.rotation_range = rotation_range
        self.gt_list = []
        self.labels_list = []
        self.prelabels_list = []

    def get_ic_scale(self, ratio):
        new_ratio = random.uniform(ratio - 0.1, ratio + 0.1)
        org_size = self.background_ref.size[0]
        size = int(np.random.uniform(org_size * new_ratio, org_size * new_ratio))
        return tuple((size, size))

    def get_rotation(self):
        return np.random.normal(0, self.rotation_range)

    def get_shift(self, rotated_ic):
        b_arr = np.asarray(self.background)
        ic_arr = np.asarray(rotated_ic)
        image_shift_x = np.random.randint(0, b_arr.shape[0] - ic_arr.shape[0] - 1)
        image_shift_y = np.random.randint(0, b_arr.shape[1] - ic_arr.shape[1] - 1)
        return tuple((image_shift_x, image_shift_y))

    def get_shift_corners(self, ic):
        b_arr = np.asarray(self.background)
        ic_arr = np.asarray(ic).reshape((-1, 2))

        while True:
            image_shift_x = np.random.randint(0, b_arr.shape[0] - 1)
            image_shift_y = np.random.randint(0, b_arr.shape[1] - 1)

            ic_arr += np.array((image_shift_x, image_shift_y))

            if np.all(ic_arr[:, 0] > 0) and np.all(ic_arr[:, 0] < b_arr.shape[0]) and np.all(
                    ic_arr[:, 1] < b_arr.shape[1]) and np.all(ic_arr[:, 1] > 0):
                return (image_shift_x, image_shift_y), np.reshape(ic_arr, (-1,))

            ic_arr -= np.array((image_shift_x, image_shift_y))

    def get_custom_gt_box(self):
        img_arr = np.array(self.ic)
        h, w, = img_arr.shape[0], img_arr.shape[1]
        diag = math.sqrt(h ** 2 + w ** 2)
        center = np.array([h / 2, w / 2], dtype=np.uint32)

        x_min = center[0] - int(diag / 2)
        y_min = center[1] - int(diag / 2)
        x_max = center[0] + int(diag / 2)
        y_max = center[1] + int(diag / 2)
        return [x_min, y_min, x_max, y_max]

    def is_out_of_bounds(self, coords):
        coords_mat = np.array(coords).reshape((-1, 2))
        b_arr = np.array(self.background)
        bw = b_arr.shape[0]
        bh = b_arr.shape[1]

        for corner in coords_mat:
            if np.any(corner < 0) or corner[0] > bw or corner[1] > bh:
                return True

        return False

    def place_ic(self, version):
        if version == '1:':
            rotation = self.get_rotation()
            new_ic = self.ic.resize(self.get_ic_scale(args.ratio))
            new_ic = new_ic.rotate(rotation, resample=Image.Resampling.BICUBIC, fillcolor='green', expand=True)
            shift = self.get_shift(new_ic)
            gt_new_ic = get_gt_box(new_ic)
            gt_new_ic = shift_gt_coords(gt_new_ic, shift)

            if self.is_out_of_bounds(gt_new_ic):
                return

            if len(self.gt_list) > 0:
                for gt in self.gt_list:
                    if is_overlapping(gt, gt_new_ic):
                        return

            self.gt_list.append(gt_new_ic)
            self.background.paste(new_ic, shift)
        else:
            rotation = self.get_rotation()
            new_ic = self.ic.resize(self.get_ic_scale(args.ratio))
            ic_w, ic_h = new_ic.size
            new_ic = new_ic.rotate(rotation, resample=Image.Resampling.BICUBIC, fillcolor='green', expand=True)
            shift = self.get_shift(new_ic)
            gt_new_ic = get_gt_box(new_ic)
            gt_new_ic = shift_gt_coords(gt_new_ic, shift)
            ic_center = get_rotation_center(new_ic)
            prelabels = center_to_corners(ic_center, ic_w, ic_h, version=version)
            prelabels = shift_ic_corners(prelabels, shift)
            prelabels = rotate_coords(prelabels, rotation)

            if self.is_out_of_bounds(gt_new_ic):
                return

            if len(self.gt_list) > 0:
                for gt in self.gt_list:
                    if is_overlapping(gt, gt_new_ic):
                        return

            self.prelabels_list.append(prelabels)
            self.gt_list.append(gt_new_ic)
            self.background.paste(new_ic, shift)

    def draw_bb(self):
        for gt in self.gt_list:
            self.background = cv2.rectangle(np.ascontiguousarray(self.background, dtype=np.uint8), (gt[0], gt[1]),
                                            (gt[2], gt[3]), (0, 0, 255), 1)

    def draw_circle(self):
        for gt in self.prelabels_list:
            for corner in np.array(gt).reshape((-1, 2)):
                self.background = cv2.circle(np.ascontiguousarray(self.background, dtype=np.uint8),
                                             tuple(map(int, corner)), 3, (0, 0, 255), 3)

    def get_final_image(self):
        final = np.ascontiguousarray(self.background, dtype=np.uint8)
        self.background = self.background_ref.copy()
        return final

    def generate_image(self, version, bb=False, circle=False, show=False):
        assert version == '2' or version == '1', 'Dataset must be either 1 or 2 but instead the chosen type is {v}'.format(
            v=version)

        count = np.random.randint(max(0, self.ic_count - 2), self.ic_count + 2)

        while len(self.gt_list) < count:
            self.place_ic(version=version)

        if version == '2':
            self.gt_list.clear()

        if bb:
            self.draw_bb()

        if circle and version == '2':
            self.draw_circle()

        if show:
            Image.fromarray(self.background).show()

        return self.get_final_image()

    def generate_dataset(self, parent_dir=args.parent_dir,
                         size=1000, version='1'):
        print('Generating Images and Labels...')
        for i in tqdm(range(size)):
            new_img = self.generate_image(version=version)
            label = 'PCB_{ic_count}_{rotation_range}_{order}.jpg'.format(ic_count=self.ic_count,
                                                                         rotation_range=self.rotation_range, order=i)
            cv2.imwrite(os.path.join(parent_dir, label), new_img)
            self.gt_list_to_label(version, name=label)

        self.save_labels(parent_dir, version)

    def gt_list_to_label(self, version, name, ic=1):
        if version == '1':
            for gt in self.gt_list:
                label = [name, int(gt[0]), int(gt[2]), int(gt[1]), int(gt[3]), ic]
                self.labels_list.append(label)
            self.gt_list.clear()
        else:
            for prelabel in self.prelabels_list:
                label = [name, *map(int, prelabel), ic]
                self.labels_list.append(label)
            self.prelabels_list.clear()

    def save_labels(self, parent_dir, version):
        print('Saving labels...')
        self.labels_list = np.array(self.labels_list)
        DF = pd.DataFrame(self.labels_list)
        if version == '1':
            header = ['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id']
        else:
            header = ['frame', 'tl_x', 'tl_y', 'tr_x', 'tr_y', 'bl_x', 'bl_y', 'br_x', 'br_y', 'cx', 'cy', 'class_id']

        print('Label Format: ', header)

        DF.to_csv(os.path.join(parent_dir, 'labels.csv'), header=header, index=False)


data_generator = SSD_Data_Generator('U26_ic.png', 'green_background.png', rotation_range=args.rotation_range,
                                    ic_count=args.ic_count,
                                    size=(args.size, args.size))

data_generator.generate_dataset(size=args.dataset_size, version=args.version)

split_labels(args.parent_dir)
