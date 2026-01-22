import numpy as np
import time
import matplotlib.pyplot as plt

def generate_data(n, d):
    X = np.random.randn(n, d)
    true_w = np.random.randn(d)
    y = X @ true_w + np.random.randn(n) * 0.1
    return X, y

def f(w, X, y):
    y_pred = X @ w
    return np.mean((y - y_pred) ** 2)

def df(w, X, y):
    n = len(y)
    y_pred = X @ w
    return (-2 / n) * X.T @ (y - y_pred)

def gradient_descent(X, y, lr=0.1, tolerance=1e-6, max_epochs=5000):
    n, d = X.shape
    w = np.zeros(d)
    prev_loss = float('inf')

    epoch = 0
    while epoch < max_epochs:
        loss = f(w, X, y)

        if abs(prev_loss - loss) < tolerance:
            return epoch, loss

        grad = df(w, X, y)
        w = w - lr * grad

        prev_loss = loss
        epoch = epoch + 1

    loss = f(w, X, y)
    return max_epochs, loss

def plot_graphs(x_vals, y_vals, x_label, y_label, title):
    plt.plot(x_vals, y_vals, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()

def experiment_1():
    sizes = [100, 500, 1000, 5000, 10000]
    iterations = []

    for n in sizes:
        X, y = generate_data(n, d=5)
        iters, _ = gradient_descent(X, y)
        iterations.append(iters)

    plot_graphs(
        sizes,
        iterations,
        "Number of Data Points",
        "Iterations to Converge",
        "Iterations to Converge vs Number of Data Points"
    )

def experiment_2():
    dimensions = [1, 5, 10, 20, 50, 100]
    times = []

    n_fixed = 5000
    lr = 0.1
    tolerance = 1e-6
    max_epochs = 5000

    for d in dimensions:
        X, y = generate_data(n_fixed, d=d)

        start = time.perf_counter()
        gradient_descent(X, y, lr=lr, tolerance=tolerance, max_epochs=max_epochs)
        end = time.perf_counter()

        times.append(end - start)

    plot_graphs(
        dimensions,
        times,
        "Number of Features",
        "Time to Converge (seconds)",
        "Time to Converge vs Number of Features"
    )

if __name__ == "__main__":
    experiment_1()
    experiment_2()
