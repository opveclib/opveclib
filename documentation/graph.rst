Graph
=====
This example demonstrates that ovl can express graph data structures and proces them. In this example we implemented a
method for triangle counting in an undirected graph. Assume the graph is given by

.. math::
   G = (V, E)

where :math:`V` is a set of vertex ids and :math:`E` is a set with vertex tuples indicating the from-to vertex edge.

Example
-------
To illustrate the definitions we specify a simple graph with five vertices and seven edges.

.. math::
   V = \{0, 1, 2, 3, 4\}
.. math::
   E = \{(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 4), (3, 4)\}

This graph has three triangles (0, 1, 3), (0, 2, 3), and (2, 3, 4).

Triangle Counting
-----------------
There are edge-based and vertex-based algorithms for triangle counting. For the GPU edge based algorithms are better
suited because they contain more parallelism. The idea of the edge-based algorithm for triangle counting is:

Set a triangle counter to zero.
Take each edge (u, v) in the graph

1. Compute the intersection of the neighbors of u with the neighbors of v.

2. Add the cardinality of the intersection set to the triangle counter.

Now, for an undirected graph this triangle counter contains three times the count of the triangles.

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~
* To avoid the tripple count of triangles, we enforce an edge order, e.g. (u, v) only exists in the set E if u < v.
* For the described algorithm it is efficient to have a flattened list of lists representation.
* To effciently compute the intersection between two sub-sets we store all neighboring lists sorted.

Testing
-------
As test case we download the graph https://snap.stanford.edu/data/ca-HepTh.html which has 28339 triangles. This graph
unzipped and stored in your temporary directory. We load the edge list representation and transform it into an adjacency
list representation using a single flattened list. This flatten list representation is the input to our triangle
counting methods. We count triangles using three implementations of the same algorithm.

1. Implementation in OVL and compilation for the GPU.

2. Implementation in OVL and compilation for the CPU.

3. Implementation in numpy and execution on the CPU.
