import os
import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


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
    return


def corners_to_center(coords):
    coords = np.asarray(coords)
    w = int(abs(coords[2] - coords[0]))
    h = int(abs(coords[3] - coords[1]))
    center_x = int(coords[0] + w / 2)
    center_y = int(coords[1] + h / 2)
    return [center_x, center_y, w, h]


def rotate_coords(coords, angle):
    old = np.asarray(coords)

    theta = -angle
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    corner_mat = np.reshape(old, (-1, 2))
    corner_mat = R.dot(corner_mat.T)
    corner_mat = corner_mat.T
    new_corners = np.reshape(corner_mat, (1,))
    return new_corners


def shift_coords(coords, shift):
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
    img_arr = np.array(ic)
    h, w, = img_arr.shape[0], img_arr.shape[1]
    x_min = 0
    y_min = 0
    x_max = x_min + w
    y_max = y_min + h
    return [x_min, y_min, x_max, y_max]


def is_overlapping(coords1, coords2):
    if coords1[0] > coords2[2] or coords2[0] > coords1[2]:
        return False

    if coords1[1] > coords2[3] or coords2[1] > coords1[3]:
        return False

    return True


def split_labels(labels_parentdir, split_ratio=0.7):
    print('Creating training and validation split...')
    assert 1.0 > split_ratio > 0, 'Ratio should be between 0 and 1'
    all_labels = pd.read_csv(os.path.join(labels_parentdir, "labels_train.csv"))
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


class SSD_Data_Generator():
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

    def get_ic_scale(self):
        org_size = self.background_ref.size[0]
        size = int(np.random.uniform(org_size * 0.1, org_size * 0.4))
        return tuple((size, size))

    def get_rotation(self):
        return np.random.normal(0, self.rotation_range)

    def get_shift(self, rotated_ic):
        b_arr = np.array(self.background)
        ic_arr = np.array(rotated_ic)
        image_shift_x = np.random.randint(0, b_arr.shape[0] - ic_arr.shape[0] - 1)
        image_shift_y = np.random.randint(0, b_arr.shape[1] - ic_arr.shape[1] - 1)
        return tuple((image_shift_x, image_shift_y))

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
        b_arr = np.array(self.background)
        bw = b_arr.shape[0]
        bh = b_arr.shape[1]

        return coords[0] < 0 or coords[1] < 0 or coords[2] > bw or coords[3] > bh

    def place_ic(self):
        rotation = self.get_rotation()
        new_ic = self.ic.resize(self.get_ic_scale())
        new_ic = new_ic.rotate(rotation, resample=Image.Resampling.BICUBIC, fillcolor='green', expand=True)
        shift = self.get_shift(new_ic)
        gt_new_ic = get_gt_box(new_ic)
        gt_new_ic = shift_coords(gt_new_ic, shift)

        if self.is_out_of_bounds(gt_new_ic):
            return

        if len(self.gt_list) > 0:
            for gt in self.gt_list:
                if is_overlapping(gt, gt_new_ic):
                    return

        self.gt_list.append(gt_new_ic)
        self.background.paste(new_ic, shift)

    def draw_bb(self):
        for gt in self.gt_list:
            self.background = cv2.rectangle(np.ascontiguousarray(self.background, dtype=np.uint8), (gt[0], gt[1]),
                                            (gt[2], gt[3]), (0, 0, 255), 1)

    def get_final_image(self, size=(300, 300)):
        final = np.ascontiguousarray(self.background, dtype=np.uint8)
        self.background = self.background_ref.copy()
        return final

    def generate_image(self, bb=True, show=False):
        while len(self.gt_list) < self.ic_count:
            self.place_ic()

        if bb:
            self.draw_bb()

        if show:
            Image.fromarray(self.background).show()

        return self.get_final_image()

    def generate_dataset(self, parent_dir='/home/token/AOI/datasets/self_generated/',
                         size=1000, split='train'):
        print('Generating Images and Labels...')
        for i in tqdm(range(size)):
            new_img = self.generate_image()
            label = 'PCB_{ic_count}_{rotation_range}_{order}.jpg'.format(ic_count=self.ic_count,
                                                                         rotation_range=self.rotation_range, order=i)
            cv2.imwrite(os.path.join(parent_dir, label), new_img)
            self.gt_list_to_label(name=label)

        self.save_labels(parent_dir, split)

    def gt_list_to_label(self, name, ic=1):
        for gt in self.gt_list:
            label = [name, gt[0], gt[2], gt[1], gt[3], ic]
            self.labels_list.append(label)

        self.gt_list.clear()

    def save_labels(self, parent_dir, split):
        print('Saving labels...')
        self.labels_list = np.array(self.labels_list)
        DF = pd.DataFrame(self.labels_list)
        header = ['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id']
        print('Label Format: ', header)

        assert split == 'train' or split == 'val', 'Wrong split type. Must be: train, val or trainval'

        if split == 'train':
            DF.to_csv(os.path.join(parent_dir, 'labels_train.csv'), header=header, index=False)
        else:
            DF.to_csv(os.path.join(parent_dir, 'labels_val.csv'), header=header, index=False)


# data_generator = SSD_Data_Generator('U26_ic.png', 'green_background.png', rotation_range=20, ic_count=4,
#                                    size=(256, 256))
# data_generator.generate_dataset(size=5000)

# split_labels('/home/token/AOI/datasets/self_generated/')

image_resize(img_path='/home/token/Downloads/IMG_4233.jpg', size=(256, 256), save_path='/home/token/Downloads/PCB_256.jpg')
