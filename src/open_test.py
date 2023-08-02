# import cv2
# import torch
# import os
# from PIL import Image, ImageOps
# from torchvision import transforms
# from torch import nn
#
#
# class EMNISTCNN(nn.Module):
#     def __init__(self, fmaps1, fmaps2, dense, dropout):
#         super(EMNISTCNN, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=fmaps1, kernel_size=5, stride=1, padding=2),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=fmaps1, out_channels=fmaps2, kernel_size=5, stride=1, padding=2),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         self.fcon1 = nn.Sequential(nn.Linear(7 * 7 * fmaps2, dense), nn.LeakyReLU())
#         self.fcon2 = nn.Linear(dense, 27)
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(self.fcon1(x))
#         x = self.fcon2(x)
#         return x
#
#
# # Create the model instance
# model = EMNISTCNN(fmaps1=40, fmaps2=160, dense=200, dropout=0.5)
#
# # Load the pre-trained weights
# checkpoint = torch.load('/Users/maorazriel/PycharmProjects/pythonProject4/src/torch_emnistcnn.pt', map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint)
#
# # Preprocess an input image
# transform = transforms.Compose([
#     transforms.Grayscale(),         # Convert the image to grayscale
#     transforms.Resize((28, 28)),    # Resize the image to 28x28 pixels
#     transforms.ToTensor(),          # Convert the image to a PyTorch tensor
#     transforms.Normalize((0.5,), (0.5,))  # Normalize the image
# ])
#
# # Load the larger image and convert to grayscale
# large_img = cv2.imread('/Users/maorazriel/PycharmProjects/pythonProject4/Images/try.jpeg', cv2.IMREAD_GRAYSCALE)
#
# # Threshold the image to binary for contour detection
# _, binary_img = cv2.threshold(large_img, 127, 255, cv2.THRESH_BINARY_INV)
#
# # Find contours in the image
# contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Create a directory to hold the images if it doesn't exist
# if not os.path.exists('letters'):
#     os.makedirs('letters')
#
# # Open a file to store the labels
# labels_file = open('labels.txt', 'w')
#
# # For each contour...
# for idx, contour in enumerate(contours):
#     # Get the bounding rectangle
#     x, y, w, h = cv2.boundingRect(contour)
#
#     # Crop the original image to this rectangle
#     cropped_img = large_img[y:y+h, x:x+w]
#
#     # Convert the cropped image to a PIL Image and resize it
#     pil_img = Image.fromarray(cropped_img).resize((28, 28))
#
#     # Invert the colors
#     pil_img = ImageOps.invert(pil_img)
#
#     # Save the image in the directory
#     pil_img.save('letters/letter_{}.png'.format(idx))
#
#     # Preprocess the image
#     img = transform(pil_img).unsqueeze(0)
#
#     # Make sure we are in evaluation mode
#     model.eval()
#
#     # Run the model
#     with torch.no_grad():
#         output = model(img)
#
#     _, predicted = torch.max(output, 1)
#     predicted_label = chr(predicted.item() + ord('a') - 1)
#     print('Predicted class:', predicted_label)  # Convert class index to a lowercase letter
#
#     # Write the label to the labels file
#     labels_file.write('letter_{}: {}\n'.format(idx, predicted_label))
#
# # Close the labels file
# labels_file.close()


import cv2
import torch
import os
from PIL import Image, ImageOps
from torchvision import transforms
from torch import nn


class EMNISTCNN(nn.Module):
    def __init__(self, fmaps1, fmaps2, dense, dropout):
        super(EMNISTCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=fmaps1, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=fmaps1, out_channels=fmaps2, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fcon1 = nn.Sequential(nn.Linear(7 * 7 * fmaps2, dense), nn.LeakyReLU())
        self.fcon2 = nn.Linear(dense, 27)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fcon1(x))
        x = self.fcon2(x)
        return x


# Create the model instance
model = EMNISTCNN(fmaps1=40, fmaps2=160, dense=200, dropout=0.5)

# Load the pre-trained weights
checkpoint = torch.load('/Users/maorazriel/PycharmProjects/pythonProject4/src/torch_emnistcnn.pt',
                        map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

# Preprocess an input image
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert the image to grayscale
    transforms.Resize((28, 28)),  # Resize the image to 28x28 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image
])

# Load the larger image and convert to grayscale
large_img = cv2.imread('/Users/maorazriel/PycharmProjects/pythonProject4/Images/img_4.png', cv2.IMREAD_GRAYSCALE)

# Threshold the image to binary for contour detection
_, binary_img = cv2.threshold(large_img, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# For visualization
large_img_rgb = cv2.cvtColor(large_img, cv2.COLOR_GRAY2RGB)

# Create a directory to hold the images if it doesn't exist
if not os.path.exists('letters'):
    os.makedirs('letters')

# Open a file to store the labels
labels_file = open('labels.txt', 'w')

# For each contour...
for idx, contour in enumerate(contours):
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)

    # Draw the bounding rectangle on the RGB image
    cv2.rectangle(large_img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    # Crop the original image to this rectangle
    cropped_img = large_img[y-2:y + h+5, x:x + w]

    # Convert the cropped image to a PIL Image and resize it
    pil_img = Image.fromarray(cropped_img).resize((28, 28))

    # Invert the colors
    pil_img = ImageOps.invert(pil_img)

    # Save the image in the directory
    pil_img.save('letters/letter_{}.png'.format(idx))

    # Preprocess the image
    img = transform(pil_img).unsqueeze(0)

    # Make sure we are in evaluation mode
    model.eval()

    # Run the model
    with torch.no_grad():
        output = model(img)

    _, predicted = torch.max(output, 1)
    predicted_label = chr(predicted.item() + ord('a') - 1)
    print('Predicted class:', predicted_label)  # Convert class index to a lowercase letter

    # Write the label to the labels file
    labels_file.write('letter_{}: {}\n'.format(idx, predicted_label))

# Close the labels file
labels_file.close()

# Show the image with bounding rectangles
cv2.imshow('Image with Bounding Rectangles', large_img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
