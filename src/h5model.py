from keras.models import model_from_json
import cv2
import os
from PIL import Image, ImageOps
import numpy as np


# Load the model
json_file = open('/Users/maorazriel/PycharmProjects/pythonProject4/src/model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/Users/maorazriel/PycharmProjects/pythonProject4/src/model/model.h5")

# Define a function to preprocess the image
def preprocess_image(img):
    # Resize image to 28x28
    img = cv2.resize(img, (28, 28))

    # Invert the image colors
    img = 255 - img

    # Normalize the image pixels to be between 0 and 1
    img = img.astype('float32') / 255

    # Flatten the image
    img = img.reshape(-1, 784)

    return img

# Load the larger image and convert to grayscale
large_img = cv2.imread('/Users/maorazriel/PycharmProjects/pythonProject4/Images/try.jpeg', cv2.IMREAD_GRAYSCALE)

# Threshold the image to binary for contour detection
_, binary_img = cv2.threshold(large_img, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a directory to hold the images if it doesn't exist
if not os.path.exists('letters'):
    os.makedirs('letters')

# Open a file to store the labels
labels_file = open('labels.txt', 'w')

# For each contour...
for idx, contour in enumerate(contours):
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the original image to this rectangle
    cropped_img = large_img[y:y+h, x:x+w]

    # Preprocess the image
    img = preprocess_image(cropped_img)

    # Run the model
    predictions = loaded_model.predict(img)
    predicted_label = np.argmax(predictions, axis=-1)
    predicted_label = chr(predicted_label[0] + ord('a') - 1)
    print('Predicted class:', predicted_label)  # Convert class index to a lowercase letter

    # Write the label to the labels file
    labels_file.write('letter_{}: {}\n'.format(idx, predicted_label))

# Close the labels file
labels_file.close()
