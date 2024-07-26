import random
import numpy as np

inputs = [10, 10]
goal = inputs[0] + inputs[1]


def TEST_CASE():
    weights = [np.float64(0.7538583476227256), np.float64(1.2461416472535665)]
    out = calculate_output(inputs, weights)
    print(out)


def calculate_output(inputs_, weights_):
    return np.dot(weights_, inputs_)


def main():
    weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
    learning_rate = 0.0001

    for i in range(1000):
        out = calculate_output(inputs, weights)
        error = goal - out

        weights[0] += learning_rate * error * inputs[0]
        weights[1] += learning_rate * error * inputs[1]

        print(f"Iteration {i + 1}, Weights: {weights}, Output: {out}, Error: {error}")

        if abs(error) < 1e-6:
            break


if __name__ == "__main__":
    # main()
    TEST_CASE()
