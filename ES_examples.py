from evolutionary_strategy import Evolutionary_algorithm, Solver_Params
import cec2017.functions as functions
import matplotlib.pyplot as plt

first = [
    [10, 2],
    [22, 3],
    [3, 1],
    [45, 5],
    [5, 7],
    [6, 2],
    [7, 3],
    [8, 1],
    [1, 2],
    [2, 3],
    [3, 1],
    [4, 5],
    [5, 7],
    [63, 2],
    [7, 3],
]


def f1x(x):
    f1 = functions.f1
    return f1([x])


test = Evolutionary_algorithm(f1x)
params = Solver_Params(first, 0.55, 1000, 0.6)
results = test.evolutionary_algorithm(params)
xv, yv = (
    [i for i, j in results.all_xmin],
    [j for i, j in results.all_xmin],
)
plt.plot(xv, yv, "go", label="starting point")
print(results.xmin)
print(len(results.all_xmin))
plt.show()
