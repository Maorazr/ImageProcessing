import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import torch
import os
from PIL import Image, ImageOps
from torchvision import transforms
from torch import nn

global_rectangles = []
selected_indices = []

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


model = EMNISTCNN(fmaps1=40, fmaps2=160, dense=200, dropout=0.5)

# Load the pre-trained weights
checkpoint = torch.load('/Users/maorazriel/PycharmProjects/pythonProject4/src/torch_emnistcnn_over.pt',
                        map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

transform = transforms.Compose([
    transforms.Grayscale(),  # Convert the image to grayscale
    transforms.Resize((28, 28)),  # Resize the image to 28x28 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image
])


def get_rectangle_from_word(word_to_rect, target_word):
    if target_word in word_to_rect:
        return word_to_rect[target_word]
    else:
        return None


def assign_characters_to_words(word_rects, char_rects):
    char_to_word = {}  # dictionary with char as key and corresponding word as value

    for char in char_rects:
        for word in word_rects:
            # Check if the character rectangle is completely contained within the word rectangle
            if word[0] <= char[0] and word[1] <= char[1] and word[0] + word[2] >= char[0] + char[2] and \
                    word[1] + word[3] >= char[1] + char[3]:
                char_to_word[tuple(char)] = tuple(word)
                break  # If a character can be assigned to a word, we break the loop

    return char_to_word


def openImageFindContours():
    image_file_name = input("Please enter the image file name: ")
    large_img = cv2.imread(f'/Users/maorazriel/PycharmProjects/pythonProject4/Images/{image_file_name}',
                           cv2.IMREAD_GRAYSCALE)

    _, binary_img = cv2.threshold(large_img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, large_img


def find_letters_rect(contours):
    rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rects.append([x, y, x + w, y + h])

    rects = np.array(rects)
    return rects


def find_word_rects(letter_rects):
    global global_rectangles
    rectangles = []
    clustering = DBSCAN(eps=60, min_samples=1).fit(letter_rects)
    for idx, class_ in enumerate(set(clustering.labels_)):
        if class_ != -1:
            same_group = np.array(letter_rects)[np.where(clustering.labels_ == class_)[0]]
            x_min = np.min(same_group[:, 0])
            y_min = np.min(same_group[:, 1])
            x_max = np.max(same_group[:, 2])
            y_max = np.max(same_group[:, 3])
            rectangles.append((x_min, y_min, x_max, y_max))
    global_rectangles = rectangles
    return rectangles



def find_rectangle(point):

    rectangles = global_rectangles
    x, y = point
    for idx, rect in enumerate(rectangles):
        x_min, y_min, x_max, y_max = rect
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return rect, idx
    return None, None


def click_and_crop(event, x, y, flags, param):
    global global_rectangles, selected_indices
    image = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        rect, rect_index = find_rectangle((x, y))

        if rect is not None and rect_index not in selected_indices:
            selected_indices.append(rect_index)
            cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.imshow("image", image)


def combine_letters_into_words(word_to_chars, char_to_image):
    word_to_string = {}
    for word_rect, char_rects in word_to_chars.items():
        word = ''.join(char_to_image[char_rect] for char_rect in char_rects)
        word_to_string[word_rect] = word
    return word_to_string


def crop_letters_from_image(contours, large_img):
    dict = {}
    letters = []
    for idx, contour in enumerate(contours):
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        rect = [x, y, x + w, y + h]
        letters.append(rect)
        print(x, y, w, h)
        margin = 1

        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(large_img.shape[1], x + w + margin)
        y_end = min(large_img.shape[0], y + h + margin)
        cropped_img = large_img[y_start:y_end, x_start:x_end]
        # cropped_img = cropped_img.astype(np.uint8)
        pil_img = Image.fromarray(cropped_img).resize((28, 28))
        pil_img = ImageOps.invert(pil_img)
        pil_img.save('letters/letter_{}.png'.format(idx))
        img = transform(pil_img).unsqueeze(0)
        model.eval()

        # Run the model
        with torch.no_grad():
            output = model(img)

        _, predicted = torch.max(output, 1)
        predicted_label = chr(predicted.item() + ord('a') - 1)
        dict[tuple(rect)] = predicted_label

    return dict


# def generate_words(letter_to_word, letter_to_char):
#     # Create a dictionary to hold the letters for each word
#     word_to_letters = {}
#     for letter_rect, word_rect in letter_to_word.items():
#         if word_rect in word_to_letters:
#             word_to_letters[word_rect].append((letter_rect, letter_to_char[letter_rect]))
#         else:
#             word_to_letters[word_rect] = [(letter_rect, letter_to_char[letter_rect])]
#
#     # Now sort the letters within each word by their x coordinate
#     for word_rect, letters in word_to_letters.items():
#         letters.sort(key=lambda x: x[0][0])
#
#     # Concatenate the sorted letters to form words and map the words to their word rectangle
#     string_to_word = {''.join([letter[1] for letter in letters]): word_rect for word_rect, letters in
#                       word_to_letters.items()}
#
#     return string_to_word

def generate_words(letter_to_word, letter_to_char):
    # Create a dictionary to hold the letters for each word
    word_to_letters = {}
    for letter_rect, word_rect in letter_to_word.items():
        if word_rect in word_to_letters:
            word_to_letters[word_rect].append((letter_rect, letter_to_char[letter_rect]))
        else:
            word_to_letters[word_rect] = [(letter_rect, letter_to_char[letter_rect])]

    # Now sort the letters within each word by their x coordinate
    for word_rect, letters in word_to_letters.items():
        letters.sort(key=lambda x: x[0][0])

    # Concatenate the sorted letters to form words
    word_to_string = {word_rect: ''.join([letter[1] for letter in letters]) for word_rect, letters in
                      word_to_letters.items()}

    return word_to_string


def find_word_containing_char(word_rects, char_rect):
    word_rect_distances = []
    for word_rect in word_rects:
        if word_rect[0] <= char_rect[0] and word_rect[1] <= char_rect[1] and \
                word_rect[2] >= char_rect[2] and word_rect[3] >= char_rect[3]:
            # Calculate the Euclidean distance from the center of the char_rect to the center of the word_rect
            word_center = ((word_rect[2] - word_rect[0]) / 2, (word_rect[3] - word_rect[1]) / 2)
            char_center = ((char_rect[2] - char_rect[0]) / 2, (char_rect[3] - char_rect[1]) / 2)
            distance = ((word_center[0] - char_center[0]) ** 2 + (word_center[1] - char_center[1]) ** 2) ** 0.5
            word_rect_distances.append((distance, word_rect))

    # Sort the distances in ascending order and return the word_rect with the smallest distance
    word_rect_distances.sort()
    return word_rect_distances[0][1] if word_rect_distances else None


def map_rects_to_words(word_rects, letter_rects):
    rect_to_word = {}
    for letter_rect in letter_rects:
        word_rect = find_word_containing_char(word_rects, letter_rect)
        if word_rect is not None:
            rect_to_word[tuple(letter_rect)] = word_rect
    return rect_to_word


def find_matching_rect(word_to_rect, target_string):
    matching_rects = [rect for rect, word in word_to_rect.items() if word == target_string]

    if not matching_rects:
        return None  # No matching rectangle found

    # Sort the rectangles by the y-axis in descending order and x-axis in ascending order.
    # matching_rects.sort(key=lambda rect: (-rect[1], rect[0]))

    return matching_rects[0]  # Return the first rectangle in the sorted list


def reverse_dict_order(input_dict):
    keys = list(input_dict.keys())
    values = list(input_dict.values())
    reversed_dict = dict(zip(keys[::-1], values[::-1]))
    return reversed_dict


def gui(image, dict):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop, param=(image,))

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):  # if the 'c' key is pressed, break from the loop
            break

    if key == ord("c"):
        print("Enter your text:")
        text = input()
        words = text.split()

        for word in words:
            rect = find_matching_rect(dict, word)
            x_min, y_min, x_max, y_max = rect
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()