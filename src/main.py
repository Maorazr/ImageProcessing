from sklearn.cluster import DBSCAN
from autocorrect import Speller
from pre_processing_to_EMNIST import *
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow.compat.v2 as tf
from keras import backend as K
from keras.models import model_from_json
import warnings
import cv2
import torch
import time
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches
import random
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches




def main(img_path):
    parameters = {
                "pixels_from_edge": 7,
                "time_between_frames": 2,
                "min_area": 18000,
                "min_lets": 40,
                "eps_lets": 5,
                "min_words": 15,
                "eps_words": 31,
                "sigma": 0,
                "x_pad": 3,
                "i_dot_thresh": 35,
                "Thresh": 150,
                "y_pad": 3,
    }

    Thresh = parameters["Thresh"]
    img = cv2.imread(img_path)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    # kernel = np.ones((5, 5), np.uint8)
    # erosion = cv2.erode(dst, kernel, iterations=1)
    gray_frame = cv2.cvtColor(dst , cv2.COLOR_BGR2GRAY)
    #gray_frame = find_biggest_rect(gray_frame, parameters)
    _, binary = cv2.threshold(gray_frame, Thresh, 255, cv2.THRESH_BINARY)  # apply thresholding

    min_words = parameters.get("min_words")
    eps_words = parameters.get("eps_words")

    find_sequence((255 - binary) / 255, min_words, eps_words)


def find_sequence(X, min_samps_words, eps_words):
    temp_list = find_1(X)

    print(temp_list)
    db = DBSCAN(eps=120, min_samples=1).fit(temp_list)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    groups = []
    width, height = len(X[0]), len(X)

    for k, col in zip(unique_labels, colors):
        # Get the indices of the points that are part of the cluster
        indices = np.where(labels == k)[0]
        # Use the indices to get the points from temp_list
        xy = np.array(temp_list)[indices]
        # create a zeros matrix in the size of X and put 1 in the locations from xy
        temp = np.zeros((height, width))
        for i in range(len(xy)):
            temp[xy[i][0]][xy[i][1]] = 1
        groups.append((temp, (min(xy[:, 0]), max(xy[:, 0]), min(xy[:, 1]), max(xy[:, 1]))))
    # plot all the Bounding boxes of the words on the original image
    # Create a Matplotlib figure

    # Display the image on the axis
    plt.imshow(X)

    # Add each bounding box to the axis as a rectangle patch
    for box in groups[:-1]:
        ymin, ymax, xmin, xmax = box[1]
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    temp = groups.pop(find_closest_square(groups))
    temp_x, temp_y = temp
    # plt.show()
    plt.imshow(X)
    for i in range(len(groups)):
            try:
                temp_x,temp_y = find_closest_square_in_x(temp_y, groups)
                # plt.imshow(temp_x, cmap='gray')
                # plt.show()

                for j in range(len(groups)):
                    if temp_y == groups[j][1]:
                        temp_x,temp_y = groups.pop(j)
                        break
            except:
                # print("MOVED A NEW LINE")
                temp = groups.pop(find_closest_square(groups))
                temp_x,temp_y = temp



    for box in groups[:-1]:
        ymin, ymax, xmin, xmax = box[1]
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    plt.show()


if __name__ == '__main__':
    main('/Users/maorazriel/PycharmProjects/pythonProject4/src/try.jpeg')