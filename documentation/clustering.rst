Clustering
==========
We added this example to illustrate a variable stopping criterion for which we use tensorflow's while loop. This
example also shows the integration of Operator Vectorization Library (OVL) operators into a tensorflow compute graph.
In this example we use the k-means algorithm for clustering.

K-Means
-------
We assume that the initial cluster centers are given, both their number and their position in the n-D space. The k-means
algorithm iterates over two steps refining these initial cluster centers by matching them more closely to the data.

The k-means algorithm iterates over the two steps until convergence or a maximum number of steps is reached:

1. Find for each data point the closest cluster center and keep that assignment stored in an index.

2. Use the assignment index and recompute the cluster centers as the centroid of the assigned data.

Setup
-----
If you want to run the main method for clustering you have to install the following dependency, which will install
matplotlib for python 2.7.

.. code-block:: console

   sudo apt-get install python-matplotlib

If you want to install matplotlib for python 3.4 execute the following:

.. code-block:: console

   pip3 install matplotlib

If not done so, you have to install the development tools to compile matplotlib. This is done through the command:

.. code-block:: console

   sudo apt-get install python3-dev

For correct display of figures under python 3 you need to install the cairocffi package. Before compiling the package
you need to install the dependency on libffi-dev.

.. code-block:: console

   sudo apt-get install libffi-dev

The install cairocffi through the pip install for python 3.

.. code-block:: console

   pip3 install cairocffi


Source Code
-----------
.. autofunction:: examples.test_clustering.kMeansGPU

.. autofunction:: examples.test_clustering.kMeansTF


Testing
-------
Our test case generates data and initial cluster centers. We have two implementations of the k-means algorithm. One uses
OVL operators that are integrated into a tensorflow compute graph. Another uses only tensorflow operators. We call these
two implementations with our generated test data, which return estimated cluster centers. The comparison between these
estimates and the ground-truth is not straightforward because the order between estimated and ground-truth cluster
centers may be arbitrary. For this reason our compare method compares each estimated cluster center with all
ground-truth cluster centers. Whenever there is one ground-truth cluster center that has a small abolsute and relative
error we found a match. The test passes only if we found matches for all estimates.