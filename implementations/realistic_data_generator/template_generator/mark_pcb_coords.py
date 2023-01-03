import os
import cv2
import numpy as np
import argparse
import pandas as pd


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        corner_list.append((x, y))
        cv2.circle(img, (x, y), 5,
                   (255, 0, 0), 2)

        cv2.imshow('image', img)


def save_labels(coords_list, save_path, img_name):
    corner_array = np.array(coords_list).reshape((-1, 4))
    df = pd.DataFrame(corner_array)
    header = ['x_min', 'y_min', 'x_max', 'y_max']
    df.to_csv(os.path.join(save_path, '{image_name}_ic_corners.csv'.format(image_name=img_name)), header=header,
              index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path')
    args = parser.parse_args()

    save_dir, img_filename = os.path.split(args.img_path)
    img_filename, _ = img_filename.split('.')

    img = cv2.imread(args.img_path, 1)
    corner_list = []

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_labels(corner_list, save_dir, img_filename)
