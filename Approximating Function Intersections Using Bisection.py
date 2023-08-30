"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """
        # This function uses the bisection
        # method to approximate the intersection points
        def bisection(f, a, b, maxerr):
            z = (a+b)/2
            while abs(f(z)) > maxerr:
                if f(a) * f(z) < 0:
                    b = z
                else:
                    a = z
                z = (a + b) / 2
            return (a + b) / 2
        f = lambda x: f1(x) - f2(x)
        roots = []
        points = np.linspace(a,b,550) #550 sub - ranges
        for i in range (len(points)-1):
            a1 = points[i]
            b1 = points[i+1]
            if f(a1)*f(b1) < 0 : # if it false there's no root in this range
                roots.append(bisection(f,a1,b1,maxerr*0.001))
        return roots


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])
        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)
        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
