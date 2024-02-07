from oneplusone import (
    Evolutionary_algorithm,
    Solver_Parameters,
    Learning_params,
)

import matplotlib.pyplot as plt
import numpy as np
import cec2017.functions as functions

f1bench = functions.f1
f9bench = functions.f9


def test_plot(f, xin, sigma, tmax, inter):
    params = Solver_Parameters(xin, sigma, tmax, inter)
    test = Evolutionary_algorithm(f, params)
    results = test.run_algorithm()
    plt.plot(range(tmax), results.allgrades(), label=sigma)
    plt.plot(results.suc_ind(), results.successful(), "ro")


def test_averageplot(f, xin, sigma, tmax, inter):
    allgrades = [0 for i in range(tmax)]
    for i in range(10):
        params = Solver_Parameters(xin, sigma, tmax, inter)
        test = Evolutionary_algorithm(f, params)
        results = test.run_algorithm()
        for i, val in enumerate(results.allgrades):
            allgrades[i] = val + allgrades[i]
    allgrades = np.divide(allgrades, 10)
    plt.plot(range(tmax), allgrades, label=f"Sigma = {sigma}")


def test_first2(f, xin, sigma, tmax, inter):
    params = Solver_Parameters(xin, sigma, tmax, inter)
    test = Evolutionary_algorithm(f, params)
    results = test.run_algorithm()
    xexis = []
    yexis = []
    for x in results.all_steps():
        xexis.append(x[0])
        yexis.append(x[1])

    plt.plot(xexis, yexis, "ro", label=f"Sigma = {sigma}")
    xexis = []
    yexis = []
    for x in results.successful_x():
        xexis.append(x[0])
        yexis.append(x[1])

    plt.plot(xexis, yexis, "b+", label="Successful path")

    plt.plot(xin[0], xin[1], "go", label="starting point")
    plt.plot(xexis[-1], yexis[-1], "bo", label="final point")


def f1x(x):
    return f1bench([x])


def f9x(x):
    return f9bench([x])


x = [3.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.7, 2.1]
imax = 100
error = 1e-14
inter = 10

params = Learning_params(0.002, imax, error)
test_averageplot(f9x, x, 0.1, imax, inter)
plt.title("ES(1+1) value by iteration")

plt.xlabel("Iteration")
plt.ylabel("Value of q(x)")

plt.yscale("log")
plt.legend()
plt.show()
