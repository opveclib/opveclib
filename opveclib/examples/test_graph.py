# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.


import unittest
import numpy as np
import opveclib as ovl


@ovl.operator()
def graph_triangle_count(startEdge, fromVertex, toVertex):
    """Counts the triangles in an undirected graph.

    Notice that this method assumes that the graph is given as an adjacency list where all lists with vertex neighbors
    are sorted.

    The parallel algorithm uses the following strategy. We map one thread per edge, This is also called the edge-based
    iterator strategy.

    The idea behind the algorithm is:
        1. Go over all edges (u, v).
        2. The neighboring indices for vertex u are N(u) and for vertex v are N(v).
        3. Increment the triangle counter by | N(u) /\ N(v) | where /\ is the set intersection operator.

    We enforce an order on the vertices that avoids counting the same triangle three times, instead each triangle is
    counted once.

    Attributes: None.

    The array toVertex is a flattened list of lists structure, where startEdge encodes the start indices of the
    separate lists.

    :param startEdge: Indices into toVertex where edges start.
    :type startEdge: list.
    :param fromVertex: The from-vertex of each edge.
    :type fromVertex: list.
    :param toVertex: The to-vertex of each edge.
    :type toVertex: list.
    :return: Counts of triangles per edge.
    """
    iEdge           = ovl.position_in(toVertex.shape)[0]
    count           = ovl.output(toVertex.shape, ovl.uint64)
    nTriangle       = ovl.variable(0, ovl.uint64)

    iFromVertex     = ovl.variable(fromVertex[iEdge], fromVertex.dtype)
    iFromEdge       = ovl.variable(startEdge[iFromVertex], startEdge.dtype)
    iFromEdgeEnd    = ovl.variable(startEdge[iFromVertex + 1], startEdge.dtype)
    iiFromVertex    = ovl.variable(toVertex[iFromEdge], toVertex.dtype)

    iToVertex       = ovl.variable(toVertex[iEdge], toVertex.dtype)
    iToEdge         = ovl.variable(startEdge[iToVertex], startEdge.dtype)
    iToEdgeEnd      = ovl.variable(startEdge[iToVertex + 1], startEdge.dtype)
    iiToVertex      = ovl.variable(toVertex[iToEdge], toVertex.dtype)

    nMerge          = iToEdgeEnd-iToEdge + iFromEdgeEnd-iFromEdge # Maximum number of merges.

    # This construction is a work-around for simulating the function of a while loop.
    #TODO(raudies@hpe.com): Replace this construct by a while loop once it is available in ovl.
    for iMerge in ovl.arange(nMerge):
        doMerge = ovl.logical_and(iFromEdge < iFromEdgeEnd, iToEdge < iToEdgeEnd)
        doMerge = ovl.logical_and(doMerge, iiFromVertex < iToVertex)

        with ovl.if_(doMerge):

            with ovl.if_(iiFromVertex < iiToVertex):
                iFromEdge <<= iFromEdge+1
                iiFromVertex <<= toVertex[iFromEdge]

            with ovl.elif_(iiFromVertex > iiToVertex):
                iToEdge <<= iToEdge+1
                iiToVertex <<= toVertex[iToEdge]

            with ovl.else_():
                nTriangle <<= nTriangle+1
                iFromEdge <<= iFromEdge+1
                iToEdge <<= iToEdge+1
                iiFromVertex <<= toVertex[iFromEdge]
                iiToVertex <<= toVertex[iToEdge]


    #TODO(raudies@hpe.com): Use a reduction function that computes a partial or complete sum.
    count[iEdge] = nTriangle # Save the triangles for each edge.

    return count


def countTrianglesCPU(startEdge, fromVertex, toVertex):
    """Count the triangles on the CPU.

    The array toVertex is a flattened list of lists structure, where startEdge encodes the start indices of the lists.

    :param startEdge: Indices into toVertex where edges start.
    :type startEdge: list.
    :param fromVertex: The from-vertex of each edge.
    :type fromVertex: list.
    :param toVertex: The to-vertex of each edge.
    :type toVertex: list.
    :return: Triangle count of graph.

    :Examples:

    .. doctest::

        >>> from opveclib.examples.test_graph import loadGraphFromTextFile, writeExampleGraphToTextFile, countTrianglesCPU
        >>> tmpName = "/tmp/v7e20.txt"
        >>> writeExampleGraphToTextFile(tmpName)
        >>> startEdge, fromVertex, toVertex = loadGraphFromTextFile(tmpName)
        >>> countTrianglesCPU(startEdge, fromVertex, toVertex)
        3
    """
    count = ovl.evaluate(graph_triangle_count(startEdge, fromVertex, toVertex), target_language='cpp')
    return np.sum(count, axis=0, dtype=np.uint64)


def countTrianglesGPU(startEdge, fromVertex, toVertex):
    """Count the triangles on the GPU.

    The array toVertex is a flattened list of lists structure, where startEdge encodes the start indices of the lists.

    :param startEdge: Indices into toVertex where edges start.
    :type startEdge: list.
    :param fromVertex: The from-vertex of each edge.
    :type fromVertex: list.
    :param toVertex: The to-vertex of each edge.
    :type toVertex: list.
    :return: Triangle count of graph.

    :Examples:

    .. doctest::

        >>> from opveclib.examples.test_graph import loadGraphFromTextFile, writeExampleGraphToTextFile, countTrianglesGPU
        >>> tmpName = "/tmp/v7e20.txt"
        >>> writeExampleGraphToTextFile(tmpName)
        >>> startEdge, fromVertex, toVertex = loadGraphFromTextFile(tmpName)
        >>> countTrianglesGPU(startEdge, fromVertex, toVertex)
        3
    """
    count = ovl.evaluate(graph_triangle_count(startEdge, fromVertex, toVertex), target_language='cuda')
    return np.sum(count, axis=0, dtype=np.uint64)

def countTrianglesNp(startEdge, fromVertex, toVertex):
    """Count the triangles using python.

    This is a reference implementation of the operator function in python.

    The array toVertex is a flattened list of lists structure, where startEdge encodes the start indices of the lists.

    :param startEdge: Indices into toVertex where edges start.
    :type startEdge: list.
    :param fromVertex: The from-vertex of each edge.
    :type fromVertex: list.
    :param toVertex: The to-vertex of each edge.
    :type toVertex: list.
    :return: Triangle count of graph.
    """

    assert len(fromVertex)==len(toVertex), "From vertex has %d entries that must match %d entries in to vertex!" \
                                           % (len(fromVertex), len(toVertex))
    nEdge       = len(fromVertex)
    nTriangle   = 0

    for iEdge in range(nEdge):
        iFromVertex     = int(fromVertex[iEdge])
        iFromEdge       = int(startEdge[iFromVertex])
        iFromEdgeEnd    = int(startEdge[iFromVertex+1])
        iiFromVertex    = int(toVertex[iFromEdge])

        iToVertex       = int(toVertex[iEdge])
        iToEdge         = int(startEdge[iToVertex])
        iToEdgeEnd      = int(startEdge[iToVertex+1])
        iiToVertex      = int(toVertex[iToEdge])

        while ((iFromEdge < iFromEdgeEnd) & (iToEdge < iToEdgeEnd) & (iiFromVertex < iToVertex)):
            if (iiFromVertex < iiToVertex):
                iFromEdge       += 1
                iiFromVertex    = int(toVertex[iFromEdge])
            elif (iiFromVertex > iiToVertex):
                iToEdge         += 1
                iiToVertex      = int(toVertex[iToEdge])
            else:
                iFromEdge       += 1
                iToEdge         += 1
                nTriangle       += 1
                iiFromVertex    = int(toVertex[iFromEdge])
                iiToVertex      = int(toVertex[iToEdge])

    return nTriangle

def writeExampleGraphToTextFile(fileName):
    """"Writes an example graph to file.

    :param fileName: The path + name of the ascii text file to hold the edge list of the graph.
    :type fileName: String.
    """

    # The edge list of the graph with 7 vertices and 20 edges.
    edgeList = [[0,1],
                [1,2],
                [2,3],
                [3,4],
                [4,5],
                [5,6],
                [0,6],
                [1,3],
                [3,5],
                [1,6],
                [1,0],
                [2,1],
                [3,2],
                [4,3],
                [5,4],
                [6,5],
                [6,0],
                [3,1],
                [5,3],
                [6,1]]
    # Write the graph to a text file.
    with open(fileName, "w") as file:
        nEdge = len(edgeList)
        for iEdge in range(nEdge):
            file.write("%d\t%d\n" % (edgeList[iEdge][0], edgeList[iEdge][1]))



def loadGraphFromTextFile(fileName):
    """Load a graph from a text file where each line has a pair of numbers representing (from,to) of an edge.

    :param fileName: The path + name of the ascii text file to read.
    :type fileName: str.
    :return: Three lists startEdge, fromVertex, toVertex -- where startEdge contains the start/end indices of single
            lists within fromVertex and toVertex. The list fromVertex contains the from vertices and the list toVertex
            contains the list toVertex.

    The memory footprint of this adjacency list representation is (|V| + 1 + 2|E|), where |V| is the number of vertices
    and |E| is the number of edges.

    For an example of such a list representation, assume that vertices are given by v0...vn. Then the simplest
    representation is a list of lists representation such as
    index	| list entries
    v0 		| v1, v3, v7, v11, v99 			// 5 entries
    v1 		| v8, v9, v15, v32, v20, v34 	// 6 entries
    v2	    | v0, v1 						// 2 entries
    ...		| ...
    vn-1	| v3, v7, v87, v55 				// 4 entries

    We use a first array (ARRAY I) of length |V|+1 to represent the start indices for neighbor indices of a second array.
    This second array (ARRAY II) contains all the to vertex indices and is |E| long.
    Our above example has the following arrays:
    ARRAY I: for the start indices:
    -------------------------------------------------------
    | index | 0 | 1 |  2 |  3 | ... | nVertex-1 | nVertex |
    | start | 0 | 5 | 11 | 13 | ... | nEdge-4   | nEdge   |
    -------------------------------------------------------

    ARRAY II: for the TO vertices.
    -------------------------------------------------------------------------------
    | index     | 0 | 1 | 2 |  3 |  4 | 5 | 6 |  7 |  8 |  9 | 10 | 11 | 12 | ... |
    | to vertex | 1 | 3 | 7 | 11 | 99 | 8 | 9 | 15 | 32 | 20 | 34 |  0 |  1 | ... |
    -------------------------------------------------------------------------------
    """

    # ******************************************************************************************************************
    # Estimate the maximum number of edges based on the number of lines in the file.
    # ******************************************************************************************************************
    blockSize = 65536
    nEdge = 0
    with open(fileName, "rb") as fHandle:
        block = fHandle.read(blockSize)
        while block:
            nEdge += block.count(b'\n')
            block = fHandle.read(blockSize)

    # ******************************************************************************************************************
    # Read the edges from the file.
    # ******************************************************************************************************************
    edges = np.zeros((nEdge, 2), dtype=np.uint64)
    nEdge = 0
    nVertex = 0
    for line in open(fileName, 'r'):
        if len(line)==0 or line.startswith('#'):
            continue

        tokens = line.split('\t')
        assert len(tokens)==2, "Found %d tokens in line %s but expected %d tokens." % (len(tokens), line, 2)
        iFrom = int(tokens[0])
        iTo = int(tokens[1])
        edges[nEdge,0] = iFrom
        edges[nEdge,1] = iTo
        nEdge += 1
        nVertex = max(nVertex, iFrom)
        nVertex = max(nVertex, iTo)

    nVertex += 1 # Count 0th vertex index
    edges = edges[0:nEdge,:]

    # Re-index
    idToIndex = dict()
    iVertex = 0
    for iEdge in range(nEdge):
        iFrom = edges[iEdge,0]
        iTo = edges[iEdge,1]
        if iFrom in idToIndex:
            iFrom = idToIndex[iFrom]
        else:
            idToIndex[iFrom] = iVertex
            iFrom = iVertex
            iVertex += 1
        if iTo in idToIndex:
            iTo = idToIndex[iTo]
        else:
            idToIndex[iTo] = iVertex
            iTo = iVertex
            iVertex += 1
        edges[iEdge,0] = iFrom
        edges[iEdge,1] = iTo


    # ******************************************************************************************************************
    # Remove all double-linked edges.
    # ******************************************************************************************************************
    edgeExists = set() # set of tuples
    nSelf   = 0
    nDouble = 0
    iiEdge = 0
    for iEdge in range(0, nEdge):
        iFrom   = edges[iEdge,0]
        iTo     = edges[iEdge,1]
        if iFrom==iTo:
            nSelf += 1
            continue
        if iFrom > iTo:
            exists = (iFrom, iTo) in edgeExists
            if not exists:
                edgeExists.add((iFrom,iTo))
                edges[iiEdge,0] = iFrom
                edges[iiEdge,1] = iTo
                iiEdge += 1
            else:
                nDouble += 1
        else:
            exists = (iTo, iFrom) in edgeExists
            if not exists:
                edgeExists.add((iTo,iFrom))
                edges[iiEdge,0] = iTo
                edges[iiEdge,1] = iFrom
                iiEdge += 1
            else:
                nDouble += 1

    nEdge = iiEdge
    edges = edges[0:nEdge,:]

    # ******************************************************************************************************************
    # Build up the adjacency list.
    # ******************************************************************************************************************
    index = np.argsort(edges[:,0], axis=0)
    startEdge = np.zeros(nVertex+1, dtype=np.uint64)
    fromVertex = np.zeros(nEdge, dtype=np.uint64)
    toVertex = np.zeros(nEdge, dtype=np.uint64)
    iFromLast = 0
    iVertex = 1
    for iEdge in range(nEdge):
        iFrom = edges[index[iEdge],0]
        iTo = edges[index[iEdge],1]
        toVertex[iEdge] = iTo
        fromVertex[iEdge] = iFrom
        if iFrom > iFromLast:
            for iJump in range(int(iFrom-iFromLast)):
                startEdge[iVertex] = iEdge
                iVertex += 1

            iFromLast = iFrom

    while iVertex <= nVertex:
        startEdge[iVertex] = nEdge
        iVertex += 1

    # ******************************************************************************************************************
    # Sort the neighbor lists.
    # ******************************************************************************************************************
    iOffset = 0
    for iVertex in range(nVertex):
        nNeighbor = int(startEdge[iVertex+1] - startEdge[iVertex])
        toVertex[iOffset:iOffset+nNeighbor] = np.sort(toVertex[iOffset:iOffset+nNeighbor])
        iOffset += nNeighbor

    return startEdge, fromVertex, toVertex



class TestGraphTriangleCountOp(unittest.TestCase):
    """
    Test cases for the triangle counting operator.
    """
    def test(self):
        """
        This test cases compares the numpy reference implementation and the opveclib implementation with the
        ground-truth count.
        """

        # Specify the graph data.
        tmpName     = "/tmp/v7e20.txt"
        nTriangle   = 3

        writeExampleGraphToTextFile(tmpName)

        ovl.logger.debug('Testing graph %s.' % tmpName)

        startEdge, fromVertex, toVertex = loadGraphFromTextFile(tmpName)

        nTriangleNPY = countTrianglesNp(startEdge, fromVertex, toVertex)
        nTriangleCPU = countTrianglesCPU(startEdge, fromVertex, toVertex)

        assert nTriangleNPY==nTriangle
        assert nTriangleCPU==nTriangle

        if ovl.local.cuda_enabled:
            nTriangleGPU = countTrianglesGPU(startEdge, fromVertex, toVertex)
            assert nTriangleGPU==nTriangle

if __name__ == '__main__':
    ovl.clear_op_cache()
    unittest.main()
