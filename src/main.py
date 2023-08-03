from collections import OrderedDict

from utiils import *


def main():
    contours, large_img = openImageFindContours()
    letter_rects = find_letters_rect(contours)
    word_rects = find_word_rects(letter_rects)
    letter_to_word = map_rects_to_words(word_rects, letter_rects)
    dict = crop_letters_from_image(contours, large_img)
    dict2 = generate_words(letter_to_word, dict)
    dict2 = OrderedDict(reversed(list(dict2.items())))
    print(dict2)
    print(dict)
    print(letter_to_word)
    print(word_rects)
    gui(large_img, dict2)



if __name__ == '__main__':
    main()
