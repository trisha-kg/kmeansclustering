"""
Primary algorithm for k-Means clustering

This file contains the Algorithm class for performing k-means clustering.  

Trisha Guttal, tkg32
12th November 2018
"""
import math
import random
import numpy


# For accessing the previous parts of the assignment
import a6checks
import a6dataset
import a6cluster


class Algorithm(object):
    """
    A class to manage and run the k-means algorithm.

    INSTANCE ATTRIBUTES:
        _dataset [Dataset]: the dataset which this is a clustering of
        _clusters [list of Cluster]: the clusters in this clustering (not empty)
    """

    # Part A
    def __init__(self, dset, k, seeds=None):
        """
        Initializes the algorithm for the dataset ds, using k clusters.

        Parameter dset: the dataset
        Precondition: dset is an instance of Dataset

        Parameter k: the number of clusters
        Precondition: k is an int, 0 < k <= dset.getSize()

        Paramter seeds: the initial cluster indices (OPTIONAL)
        Precondition seeds is None, or a list of k valid indices into dset.
        """
        assert isinstance(dset, a6dataset.Dataset)
        self._dataset = dset
        assert type(k) == int and  0 < k <= dset.getSize()
        self._clusters = []
        assert seeds is None or a6checks.is_seed_list(seeds, k, dset.getSize()) == True
        if seeds is None:
            contentlength = len(self._dataset.getContents())
            seeds = random.sample(range(contentlength), k)
        for x in seeds:
            centroid = self._dataset.getContents()[x]
            newclusterobj = a6cluster.Cluster(self._dataset, centroid)
            self._clusters.append(newclusterobj)


    def getClusters(self):
        """
        Returns the list of clusters in this object.

        This method returns the attribute _clusters directly.  Any changes made to this
        list will modify the set of clusters.
        """
        return self._clusters


    # Part B
    def _nearest(self, point):
        """
        Returns the cluster nearest to point

        Parameter point: The point to compare.
        Precondition: point is a list of numbers (int or float), with the same dimension
        as the dataset.
        """
        assert a6checks.is_point(point) == True
        assert len(point) == self._dataset.getDimension()
        mindist = self._clusters[0].distance(point)
        nearestcluster = self._clusters[0]
        for newclusterobject in self._clusters:
            if newclusterobject.distance(point) < mindist:
                mindist = newclusterobject.distance(point)
                nearestcluster = newclusterobject
        return nearestcluster


    def _partition(self):
        """
        Repartitions the dataset so each point is in exactly one Cluster.
        """
        # First, clear each cluster of its points.  Then, for each point in the
        # dataset, find the nearest cluster and add the point to that cluster.

        for clusterobject in self._clusters:
            clusterobject.clear()
        for data in range(self._dataset.getSize()):
            clusterpoint = self._dataset.getPoint(data)
            result = self._nearest(clusterpoint)
            result.addIndex(data)


    # Part C
    def _update(self):
        """
        Returns true if all centroids are unchanged after an update; False otherwise.

        This method first updates the centroids of all clusters'.  When it is done, it
        checks whether any of them have changed. It then returns the appropriate value.
        """
        updatedclusterlist = []
        for clusterobject in self._clusters:
            updatedclusterlist.append(clusterobject.update())
        for clusterobject in updatedclusterlist:
            if clusterobject == False:
                return False
        return True


    def step(self):
        """
        Returns True if the algorithm converges after one step; False otherwise.

        This method performs one cycle of the k-means algorithm. It then checks if
        the algorithm has converged and returns the appropriate value.
        """
        # In a cycle, we partition the points and then update the means.
        self._partition()
        if self._update() == True:
            return True
        else:
            return False


    # Part D
    def run(self, maxstep):
        """
        Continues clustering until either it converges or maxstep steps
        (which ever comes first).

        Parameter maxstep: the maximum number of steps to try
        Precondition: maxstep is an int >= 0
        """
        assert type(maxstep) == int and maxstep >= 0
        for x in range(maxstep):
            if self.step() == True:
                return
