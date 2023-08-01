import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# Initialize the list of rectangles and boolean indicating
# whether cropping is being performed or not
rectangles = []
current_rectangle = []


def find_rectangle(point, rectangles):
    x, y = point
    for rect in rectangles:
        x_min, y_min, x_max, y_max = rect
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return rect
    return None


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global rectangles, current_rectangle

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        current_rectangle = [(x, y)]
        rect = find_rectangle((x, y), rectangles)

        # check if the point is within a rectangle
        if rect is not None:
            # draw a rectangle around the region of interest
            cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.imshow("image", image)


# Load the image
image = cv2.imread('/Users/maorazriel/PycharmProjects/pythonProject4/src/try.jpeg')

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
clustering = DBSCAN(eps=120, min_samples=1).fit(rects)

# Iterate over the clusters and draw bounding boxes around each one
for idx, class_ in enumerate(set(clustering.labels_)):
    if class_ != -1:
        same_group = np.array(rects)[np.where(clustering.labels_ == class_)[0]]
        # Find minimal box that contains all rectangles in the cluster
        x_min = np.min(same_group[:, 0])
        y_min = np.min(same_group[:, 1])
        x_max = np.max(same_group[:, 2])
        y_max = np.max(same_group[:, 3])

        # Add the rectangle to the list
        rectangles.append((x_min, y_min, x_max, y_max))

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# Keep the window open until the 'c' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):  # if the 'c' key is pressed, break from the loop
        break

# close all open windows
cv2.destroyAllWindows()
