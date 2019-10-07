"""
Cluster class for k-Means clustering

This file contains the class cluster.

Trisha Guttal
"""
import math
import random
import numpy


# For accessing the previous parts of the assignment
import a6checks
import a6dataset


class Cluster(object):
    """
    A class representing a cluster, a subset of the points in a dataset.

    INSTANCE ATTRIBUTES:
        _dataset [Dataset]: the dataset this cluster is a subset of
        _indices [list of int]: the indices of this cluster's points in the dataset
        _centroid [list of numbers]: the centroid of this cluster
    EXTRA INVARIANTS:
        len(_centroid) == _dataset.getDimension()
        0 <= _indices[i] < _dataset.getSize(), for all 0 <= i < len(_indices)
    """

    # Part A
    def __init__(self, dset, centroid):
        """
        Initializes a new empty cluster whose centroid is a copy of <centroid>

        Parameter dset: the dataset
        Precondition: dset is an instance of Dataset

        Parameter centroid: the cluster centroid
        Precondition: centroid is a list of ds.getDimension() numbers
        """
        assert isinstance(dset, a6dataset.Dataset)
        self._dataset = dset
        assert type(centroid) == list
        centroidcopy = centroid[:]
        self._centroid = centroidcopy
        assert len(self._centroid) == self._dataset.getDimension()
        self._indices = []


    def getCentroid(self):
        """
        Returns the centroid of this cluster.

        This getter method is to protect access to the centroid.
        """
        return self._centroid


    def getIndices(self):
        """
        Returns the indices of points in this cluster

        This method returns the attribute _indices directly.  Any changes made to this
        list will modify the cluster.
        """
        return self._indices


    def addIndex(self, index):
        """
        Adds the given dataset index to this cluster.

        If the index is already in this cluster, this method leaves the
        cluster unchanged.

        Precondition: index is a valid index into this cluster's dataset.
        That is, index is an int in the range 0.._dataset.getSize()-1.
        """
        assert type(index) == int and index >= 0
        assert index <= (self._dataset.getSize()-1)
        if not index in self._indices:
            self._indices.append(index)


    def clear(self):
        """
        Removes all points from this cluster, but leave the centroid unchanged.
        """
        self._indices = []


    def getContents(self):
        """
        Returns a new list containing copies of the points in this cluster.

        The result is a list of list of numbers.  It has to be computed from the indices.
        """
        clustercopy = []
        for x in self._indices:
            nums = self._dataset.getPoint(x)
            clustercopy.append(nums)
        return clustercopy


    # Part B
    def distance(self, point):
        """
        Returns the euclidean distance from point to this cluster's centroid.

        Parameter point: The point to be measured
        Precondition: point is a list of numbers (int or float), with the same dimension
        as the centroid.
        """
        assert a6checks.is_point(point) == True
        assert len(point) == len(self._centroid)

        pointresult = 0
        pointdim = len(point)
        total = 0
        centroidind = self.getCentroid()
        for x in range(pointdim):
            pointresult = point[x] - centroidind[x]
            square = (pointresult)**2
            total = total + square
        root = math.sqrt(total)
        return root


    def getRadius(self):
        """
        Returns the maximum distance from any point in this cluster, to the centroid.

        This method loops over the contents to find the maximum distance from
        the centroid.  If there are no points in this cluster, it returns 0.
        """
        maxdist = 0
        distlist = []
        indiceslen = len(self._indices)
        for x in self._indices:
            point = self._dataset.getContents()[x]
            pointdist = self.distance(point)
            if pointdist > maxdist:
                maxdist = pointdist
        if len(self._indices) == 0:
            maxdist = 0
        return maxdist


    def update(self):
        """
        Returns True if the centroid remains the same after recomputation; False otherwise.

        This method recomputes the _centroid attribute of this cluster. The new _centroid
        attribute is the average of the points of _contents (To average a point, average
        each coordinate separately).

        Whether the centroid "remained the same" after recomputation is determined bys
        numpy.allclose.  The return value should be interpreted as an indication of whether
        the starting centroid was a "stable" position or not.

        If there are no points in the cluster, the centroid does not change.
        """
        contentset = self.getContents()
        numrows = len(contentset)
        numcols = len(contentset[0])
        result = []
        sum = 0
        for col in range(numcols):
            for row in range(numrows):
                sum += contentset[row][col]
            result.append(sum)
            sum = 0
        averages = []
        for x in result:
            averages.append(x/len(contentset))
        avg = numpy.allclose(self._centroid, averages)

        if avg == True:
            return avg
        else:
            self._centroid = averages
            return avg


    # PROVIDED METHODS: Do not modify!
    def __str__(self):
        """
        Returns a String representation of the centroid of this cluster.
        """
        return str(self._centroid)

    def __repr__(self):
        """
        Returns an unambiguous representation of this cluster.
        """
        return str(self.__class__) + str(self)
