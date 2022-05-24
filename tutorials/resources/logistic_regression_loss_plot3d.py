from sympy import *
from sympy.plotting import plot3d
import pandas as pd

points = list(pd.read_csv("https://tinyurl.com/y2cocoo7").itertuples())

b1, b0, i, n = symbols('b1 b0 i n')
x, y = symbols('x y', cls=Function)
joint_likelihood = exp(Sum(log((1.0 / (1.0 + exp(-(b0 + b1 * x(i))))) ** y(i) \
                           * (1.0 - (1.0 / (1.0 + exp(-(b0 + b1 * x(i)))))) ** (1 - y(i))), (i, 0, n)))

# Partial derivative for m, with points substituted
d = joint_likelihood \
    .subs(n, len(points) - 1).doit() \
    .replace(x, lambda i: points[i].x) \
    .replace(y, lambda i: points[i].y)

plot3d(d)