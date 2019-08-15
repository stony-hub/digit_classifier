import sys
import numpy as np
import matplotlib.image as mpimg
from digit_classifier import DigitClassifier


def main():
    model = DigitClassifier()
    while True:
        op = input()
        if op == 'exit': break

        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        data = rgb2gray(mpimg.imread(sys.argv[1])).reshape(1, 28, 28, 1)
        res = model.predict(data)
        print(res)


if __name__ == '__main__':
    main()
