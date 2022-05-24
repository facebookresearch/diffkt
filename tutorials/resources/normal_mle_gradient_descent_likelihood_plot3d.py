from sympy import *
from sympy.plotting import plot3d
import pandas as pd
import numpy as np

points = np.random.standard_normal(30)

mu, sigma, i, n = symbols('mu sigma i n')
x = symbols('x', cls=Function)

joint_likelihood = Sum(1 / (sigma*(2*pi)**0.5) * exp(-0.5 * ((x(i)-mu) / sigma)**2), (i, 0, n)) \
    .subs(n, len(points) - 1).doit() \
    .replace(x, lambda i: points[i])

plot3d(joint_likelihood)
