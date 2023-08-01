import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# Load the image
image = cv2.imread('/Users/maorazriel/PycharmProjects/pythonProject4/src/img.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty list to store the coordinates of the bounding boxes
rects = []

# Iterate over the contours and add the bounding boxes to the list
for contour in contours:
    # Get bounding box coordinates for each contour
    x, y, w, h = cv2.boundingRect(contour)

    # Add the coordinates to the list
    rects.append([x, y, x + w, y + h])

# Convert the list of coordinates to a numpy array
rects = np.array(rects)


# Apply DBSCAN to group the coordinates
clustering = DBSCAN(eps=60, min_samples=1).fit(rects)



# Iterate over the clusters and draw bounding boxes around each one
for idx, class_ in enumerate(set(clustering.labels_)):
    print('idx = ', idx, 'class', class_)
    if class_ != -1:
        same_group = np.array(rects)[np.where(clustering.labels_ == class_)[0]]
        # Find minimal box that contains all rectangles in the cluster
        x_min = np.min(same_group[:, 0])
        y_min = np.min(same_group[:, 1])
        x_max = np.max(same_group[:, 2])
        y_max = np.max(same_group[:, 3])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
