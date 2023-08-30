"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""
from typing import Tuple

import numpy as np
import time
import random

from numpy import single
from sklearn.cluster import KMeans
import math
from shapely.geometry import Polygon
from functionUtils import AbstractShape

class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self,contour1: np.ndarray):
        """
        :param contour1:numpy.ndarray
            An array of 2D points defining the shape's contour.
        """
        self.contour1 = contour1
        pass
    def contour(self,n):
        """
        :param n:The number of points to return from the contour1 array.
        :return:numpy.ndarray
            An array of 2D points representing a subarray of the first n points from the
            contour1 array.
        """
        return self.contour1[:n]

    def area(self)-> float:
        """
        Computes and returns the area of the shape defined by the contour1 array
        :return:  float
            The area of the shape defined by the contour1 array.

        """
        n = len(self.contour1)
        area = 0
        for i in range(n - 1):
            area += self.contour1[i][0] * self.contour1[i + 1][1] - self.contour1[i][1] * self.contour1[i + 1][0]
        area += self.contour1[n - 1][0] * self.contour1[0][1] - self.contour1[n - 1][1] * self.contour1[0][0]
        return abs(area) / 2

class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """
        pass


    def shoelace(self,points):
        """
        :param points: list
        A list of tuples representing the points of the polygon, where each tuple
        contains the x and y coordinates of a point in the plane.
        :return:The area of the polygon.
        """
        n = len(points)
        area = 0
        for i in range(n - 1):
            area += points[i][0] * points[i + 1][1] - points[i][1] * points[i + 1][0]
        area += points[n - 1][0] * points[0][1] - points[n - 1][1] * points[0][0]
        return abs(area) / 2


    def area(self, contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        num_points = 1000 # num of points
        points = contour(num_points) # list that evey cell is [(x,y)]
        if len(points) <= 2: # if it less than 2 i can not calculate the area
            return np.float32(0)
        area = self.shoelace(points)
        return np.float32(area)


    def fit_shape(self, sample: callable, maxtime: float) -> Tuple[MyShape, single]:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """
        start = time.time()
        points = []
        for i in range(10000):
            if time.time() - start > maxtime +4: # check that we will have enough time to all the code
                break
            x,y = sample()
            points.append([x,y])
        if len(points) < 3: # can not continue , return Myshape
            a = MyShape(points)
            return a
        kmeans = KMeans(n_clusters=20, n_init=1) # create kmeans with 20 clusters
        kmeans.fit(points) # fit the points to the kmeanse
        cluster_centers = kmeans.cluster_centers_ # get all  centers of each cluster
        def sort_cluster_centers(cluster_centers):
            # Find the center of all centers
            center_of_centers = np.mean(cluster_centers, axis=0)

            # Calculate the angle between each center and the center of all centers
            angles = []
            for center in cluster_centers:
                delta = center - center_of_centers
                angle = math.atan2(delta[1], delta[0])
                angles.append(angle)

            # Sort the center points in descending order of the angles
            sorted_centers = [center for _, center in sorted(zip(angles, cluster_centers), reverse=True)]

            return np.array(sorted_centers)

        sorted_centers = sort_cluster_centers(cluster_centers)
        result = MyShape(sorted_centers)
        return result

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
