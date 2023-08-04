from collections import OrderedDict

from utiils import *


def main():
    gui()

    # dict2 = None
    # large_img = None
    # while True:
    #     print("Please choose an option:")
    #     print("1) Load image")
    #     print("2) Annotate by clicking the image")
    #     print("3) Annotate by typing the words")
    #     print("4) Exit")
    #     option = int(input("Your choice: "))
    #
    #     if option == 1:
    #         contours, large_img = openImageFindContours()
    #         letter_rects = find_letters_rect(contours)
    #         word_rects = find_word_rects(letter_rects)
    #         letter_to_word = map_rects_to_words(word_rects, letter_rects)
    #         dict = crop_letters_from_image(contours, large_img)
    #         dict2 = generate_words(letter_to_word, dict)
    #         dict2 = OrderedDict(reversed(list(dict2.items())))
    #     elif option == 2:
    #         if large_img is not None and dict2 is not None:
    #             cv2.namedWindow("image")
    #             cv2.setMouseCallback("image", click_and_crop, param=(large_img,))
    #             cv2.imshow("image", large_img)
    #             cv2.waitKey(0)
    #         else:
    #             print("You need to load an image first.")
    #     elif option == 3:
    #         if large_img is not None and dict2 is not None:
    #             print("Enter your text:")
    #             text = input()
    #             words = text.split()
    #             for word in words:
    #                 rect = find_matching_rect(dict2, word)
    #                 x_min, y_min, x_max, y_max = rect
    #                 cv2.rectangle(large_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    #             cv2.imshow("image", large_img)
    #             cv2.waitKey(0)
    #         else:
    #             print("You need to load an image first.")
    #     elif option == 4:
    #         cv2.destroyAllWindows()
    #         break
    #     else:
    #         print("Invalid option, please try again.")


if __name__ == '__main__':
    main()
