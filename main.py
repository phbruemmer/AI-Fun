import random
import numpy as np

inputs = [10, 9]
goal = inputs[0] + inputs[1]


def calculate_output(inputs_, weights_):
    return np.dot(weights_, inputs_)


def main():
    weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
    learning_rate = 0.01

    for i in range(1000000):
        out = calculate_output(inputs, weights)
        error = goal - out

        weights[0] += learning_rate * error * inputs[0]
        weights[1] += learning_rate * error * inputs[1]

        print(f"Iteration {i + 1}, Weights: {weights}, Output: {out}, Error: {error}")

        if abs(error) < 1e-6:
            break


if __name__ == "__main__":
    main()
