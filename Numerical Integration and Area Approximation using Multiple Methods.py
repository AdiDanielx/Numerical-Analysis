"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""
import math
from typing import Optional

import numpy as np
import time
import random

from numpy import single

from assignment2 import Assignment2


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    # implements the midpoint rule for numerical integration.
    def midpoint_rule(self, f, a, b, n):
        h = (b - a) / n
        result = 0
        result = f((a + h / 2) + n * h)
        result *= h
        return abs(result)

    # trapezoidal_rule: implements the trapezoidal
    # rule for numerical integration.
    def trapezoidal_rule(self, f: callable, a: float, b: float, n: int):
        h = (b - a) / n
        total_sum = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            x_i = a + i * h
            total_sum += f(x_i)
        return np.flost32(abs(total_sum * h))

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """
        # implements an adaptive Simpson's rule for numerical
        # integration. This is the primary method that is used
        # to calculate the definite integral of a given function.
        # n have to be even.
        def simpson(f, a, b, n):
            h = (b - a) / n
            first = f(a)
            last = f(b)
            x = a
            ssum = 0
            for i in range(n - 1):
                x += h
                value = f(x)
                if i % 2 == 0:
                    ssum += 4 * value
                else:
                    ssum += 2 * value

            total_sum = abs((h / 3) * (first + ssum + last))
            return np.float32(total_sum)

        if n == 1:
            self.midpoint_rule(f, a, b, n)
        elif n == 2:
            return self.trapezoidal_rule(f, a, b, n)
        elif n % 2 == 1:
            return simpson(f, a, b, n - 1)
        else:
            return simpson(f, a, b, n - 2)

    def areabetween(self, f1: callable, f2: callable) -> Optional[single]:
        """
        Finds the area enclosed between two functions. This method finds
        all intersection points between the two functions to work correctly.

        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area.

        In order to find the enclosed area the given functions must intersect
        in at least two points. If the functions do not intersect or intersect
        in less than two points this function returns NaN.
        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        points = list(Assignment2().intersections(f1, f2, 1, 100)) # find intersections between f1 f2
        f = lambda x: abs(f1(x) - f2(x))
        points.sort() # sort the points
        a = 0
        if len(points) < 2: # if there is less than 2 points , there is no way to calculate area
            return np.nan
        else:
            for i in range(0, len(points) - 1):
                a += abs(Assignment3.integrate(self, f, points[i], points[i + 1], 100))
            result = np.float32(a)
            return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)
        self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))
if __name__ == "__main__":
    unittest.main()
