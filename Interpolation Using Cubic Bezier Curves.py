"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random
from scipy.sparse import lil_matrix
from scipy.interpolate import interp1d


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """
        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        # Create matrix is a nXn matrix, where n is the
        # number of points used for interpolation.
        # coefficients matrix
        def get_C(n):
            C = 4 * np.identity(n, dtype=np.float64)
            np.fill_diagonal(C[1:], 1.0)
            np.fill_diagonal(C[:, 1:], 1.0)
            C[0, 0] = 2.0
            C[n - 1, n - 1] = 7.0
            C[n - 1, n - 2] = 2.0
            return C

        # Create matrix is a nX2
        # build points vector
        # The values of P[0] and P[n-1] are slightly different from the
        # rest of the entries, so they are calculated separately.
        def get_K(points, n):
            P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
            P[0] = points[0] + 2 * points[1]
            P[n - 1] = 8 * points[n - 1] + points[n]
            return P

        # Tomas solved the equation CA=P, obtained the matrices
        # C and P, and returned the matrix A
        def tomas_algorithm(C, P):
            n = len(C)
            x = [0] * n
            for i in range(n - 1):
                div = C[i + 1][i] / C[i][i]
                C[i + 1] = [C[i + 1][j] - div * C[i][j] for j in range(n)]
                P[i + 1] = P[i + 1] - div * P[i]
            x[n - 1] = P[n - 1] / C[n - 1][n - 1]
            for i in range(n - 2, -1, -1):
                x[i] = (P[i] - sum([C[i][j] * x[j] for j in range(i + 1, n)])) / C[i][i]
            return x

        # generates a list of length n with the
        # values of the b_i coefficients in the cubic polynomial
        # that interpolates the function f on each segment.
        def get_B(points, n):
            B = [0] * n
            for i in range(n - 1):
                B[i] = 2 * points[i + 1] - A[i + 1]
            B[n - 1] = (A[n - 1] + points[n]) / 2
            return B
        # The function generates a cubic
        # polynomial with given coefficients.
        def get_cubic(a, b, c, d):
            return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t,
                                                                                                              2) * c + np.power(
                t, 3) * d
        # This is a lambda function that takes
        # a real number x as input and returns
        # the corresponding value of the interpolated
        # function. It uses the values of points, A, and B
        # that were calculated in the interpolate method.
        def r(x):
            for i in range(n - 1):
                if points[i][0] <= x <= points[i + 1][0]:
                    r_X = (x - points[i][0]) / (points[i + 1][0] - points[i][0])
                    bezier = get_cubic(points[i][1], A[i], B[i], points[i + 1])
                    return bezier(r_X)[1]

        parts = list(np.linspace(a, b, n))
        points = []
        for x in parts:
            point = np.array([x, f(x)], dtype=np.float64)
            points.append(point)
        C = get_C(n - 1)  # The matrix is a (n-1)X(n-1) matrix
        P = get_K(points, n - 1)  # The matrix is a (n-1)X2
        A = tomas_algorithm(C, P)  # Solve CA=P return A
        B = get_B(points, n - 1)
        return r


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
