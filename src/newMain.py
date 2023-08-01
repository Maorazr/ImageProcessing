import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
from pre_processing_to_EMNIST import *


def find_closest_square_in_x(temp, x):
    temp_center = ((temp[0] + temp[1]) / 2, (temp[2] + temp[3]) / 2)  # Find the center of temp
    temp_x_center = temp_center[1]  # Find the x-coordinate of the center of temp
    # Find the centers and angles of all the squares in x
    x_centers = []
    angles = []
    for i in range(len(x)):
        square = x[i][1]
        x_center = ((square[0] + square[1]) / 2, (square[2] + square[3]) / 2)  # Find the center of the current square
        x_centers.append(x_center)
        # Calculate the angle between the centers of temp and the current square, in the [-pi, pi] range
        angle = np.arctan2(x_center[0] - temp_center[0], x_center[1] - temp_center[1])
        angles.append(angle)
    # Find the minimum x-distance, x-coordinate, and angle within the acceptable range
    min_x_distance = float('inf')
    min_x_center = float('inf')
    min_angle = float('inf')
    min_x_distance_index = -1
    min_x_center_index = -1
    min_angle_index = -1
    for i in range(len(x_centers)):
        # calculate the distance between the centers of temp and the current square
        x_distance = np.sqrt((x_centers[i][0] - temp_center[0]) * 2 + (x_centers[i][1] - temp_center[1]) * 2)
        # print("X Distance is : ", x_distance)
        #  print(angles[i])
        #    print(abs(abs(angles[i]) - np.pi / 2))
        if abs(angles[i]) < 0.25 or abs(angles[i]) <= min_angle:
            # check if the x of the temp center is smaller than the temp_x_center
            if (abs(temp_center[0] - x_centers[i][0]) < 25):
                if x_centers[i][1] >= temp_x_center:
                    if x_distance < min_x_distance:
                        min_x_distance = x_distance
                        min_x_center = x_centers[i][0]
                        min_angle = abs(angles[i])
                        min_x_distance_index = i
                        min_x_center_index = i
                        min_angle_index = i
    # If an acceptable x-distance, x-coordinate, and angle were found, return the corresponding square in x
    min_distance_square = min(x, key=lambda square: np.sqrt((square[0] - temp_center[0]) * 2 + (square[1] - temp_center[1]) * 2))
    return min_distance_square

def find_closest_square(groups):
    # Find the closest square to the point (0, 0)
    closest_square = min(groups, key=lambda square: np.sqrt(square[0] * 2 + square[1] * 2))
    return closest_square

def find_1(temp):
    return [(i, j) for i, row in enumerate(temp) for j, pixel in enumerate(row) if pixel == 1]

def preprocess_image(img):
    # Ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Ensure 8-bit format
    img = np.uint8(img)
    img = cv2.medianBlur(img, 5)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    img = cv2.bitwise_not(img)
    return img
# def find_biggest_rect(gray_image, parameters):
#     # Existing code...


def find_sequence(X, min_samps_words):
    X = MinMaxScaler().fit_transform(X)
    # compute DBSCAN
    # eps_words = compute_eps(X)
    db = DBSCAN(eps=4, min_samples=min_samps_words).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def compute_eps(X):
    # Compute distance matrix
    from sklearn.metrics.pairwise import euclidean_distances
    distance_matrix = euclidean_distances(X, X)

    # Compute mean distance for each point
    mean_distances = np.mean(distance_matrix, axis=1)

    # Plot mean distances
    plt.plot(sorted(mean_distances))
    plt.xlabel('Points')
    plt.ylabel('Mean Distance')
    plt.show()

    # Use knee point as eps
    kneedle = KneeLocator(range(len(mean_distances)), sorted(mean_distances), curve='convex', direction='increasing')
    return kneedle.knee_y

def main(img_path):
    parameters = {
        "pixels_from_edge": 7,
        "time_between_frames": 2,
        "min_area": 18000,
        "min_lets": 40,
        "eps_lets": 1,
        "min_words": 15,
        "eps_words": 30,
        "sigma": 0,
        "x_pad": 3,
        "i_dot_thresh": 35,
        "Thresh": 135,
        "y_pad": 3,
    }

    Thresh = parameters["Thresh"]
    img = cv2.imread(img_path)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    gray_frame = cv2.cvtColor(dst , cv2.COLOR_BGR2GRAY)
    gray_frame = preprocess_image(gray_frame)

    _, binary = cv2.threshold(gray_frame, Thresh, 255, cv2.THRESH_BINARY)  # apply thresholding

    temp_list = find_1((255 - binary) / 255)

    eps_words = compute_eps(temp_list)  # Computing optimal eps using KneeLocator
    min_words = parameters.get("min_words")

    find_sequence((255 - binary) / 255, min_words, eps_words)

if __name__ == '__main__':
    main('/Users/maorazriel/PycharmProjects/pythonProject4/src/try.jpeg')