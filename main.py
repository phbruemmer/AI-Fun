import random
import numpy as np

"""
# # #
TRY - without much knowledge about numpy and neural networks :P
# # #
"""

inputs = [2, 2]
goal = inputs[0] + inputs[1]


def calculate_output(inputs_, weights_):
    return np.dot(weights_, inputs_)


def main():
    i = 0
    weights = [random.randint(-inputs[0], inputs[1]), random.randint(-inputs[0], inputs[1])]

    while not i == 100:

        out = calculate_output(inputs, weights)

        print(i)
        if out == goal:
            i += 1
        else:
            i -= 1


if __name__ == "__main__":
    main()
