import cv2
import numpy as np
from sklearn.cluster import DBSCAN

rectangles = []
selected_indices = []


def find_rectangle(point, rectangles):
    x, y = point
    for idx, rect in enumerate(rectangles):
        x_min, y_min, x_max, y_max = rect
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return rect, idx
    return None, None


def click_and_crop(event, x, y, flags, param):
    global rectangles, selected_indices, image

    if event == cv2.EVENT_LBUTTONDOWN:
        rect, rect_index = find_rectangle((x, y), rectangles)

        if rect is not None and rect_index not in selected_indices:
            selected_indices.append(rect_index)
            cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.imshow("image", image)


image_path = '/Users/maorazriel/PycharmProjects/pythonProject4/Images/try.jpeg'
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    rects.append([x, y, x + w, y + h])


rects = np.array(rects)

clustering = DBSCAN(eps=120, min_samples=1).fit(rects)

for idx, class_ in enumerate(set(clustering.labels_)):
    if class_ != -1:
        same_group = np.array(rects)[np.where(clustering.labels_ == class_)[0]]
        x_min = np.min(same_group[:, 0])
        y_min = np.min(same_group[:, 1])
        x_max = np.max(same_group[:, 2])
        y_max = np.max(same_group[:, 3])
        rectangles.append((x_min, y_min, x_max, y_max))

# Sort rectangles from top to bottom and then from left to right
rectangles.sort(key=lambda r: (r[1], r[0]))

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):  # if the 'c' key is pressed, break from the loop
        break

if key == ord("c"):
    print("Enter your text:")
    text = input()
    words = text.split()

    if selected_indices:
        start_index = max(selected_indices) + 1
    else:
        start_index = 0

    end_index = start_index + len(words)

    for i in range(start_index, end_index):
        if i < len(rectangles):
            rect = rectangles[i]
            print(rect)
            cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()



def assign_characters_to_words(word_rects, char_rects):
    """
    Assign each character to the rectangle that represent the word that this character is part of.
    Characters are sorted according to their appropriate place in the word.

    :param word_rects: A list of 4-tuple representing the rectangles that contain words. Each tuple is (x, y, w, h).
    :param char_rects: A list of 4-tuple representing the rectangles that contain characters. Each tuple is (x, y, w, h).
    :return: A dictionary where each key is a word rectangle and each value is a list of the character rectangles that belong to that word.
    """
    word_to_chars = {word: [] for word in word_rects}

    for char in char_rects:
        for word in word_rects:
            # Check if the character rectangle is completely contained within the word rectangle
            if word[0] <= char[0] and word[1] <= char[1] and word[0] + word[2] >= char[0] + char[2] and word[1] + word[
                3] >= char[1] + char[3]:
                word_to_chars[word].append(char)
                break  # If a character can be assigned to a word, we break the loop

    for word, chars in word_to_chars.items():
        # Sort the characters by their x-coordinate
        chars.sort(key=lambda char: char[0])

    return word_to_chars
