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
from sys import _getframe
from .graph import writeExampleGraphToTextFile, loadGraphFromTextFile, countTrianglesCPU, \
    countTrianglesGPU, countTrianglesNp
import opveclib as ops

class TestGraphTriangleCountOp(unittest.TestCase):
    """
    Test cases for the triangle counting operator.
    """
    def test(self):
        """
        This test cases compares the numpy reference implementation and the opveclib implementation with the
        ground-truth count.
        """
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)

        # Specify the graph data.
        tmpName     = "/tmp/v7e20.txt"
        nTriangle   = 3

        writeExampleGraphToTextFile(tmpName)

        print('Testing graph %s.' % tmpName)

        startEdge, fromVertex, toVertex = loadGraphFromTextFile(tmpName)

        nTriangleNPY = countTrianglesNp(startEdge, fromVertex, toVertex)
        nTriangleCPU = countTrianglesCPU(startEdge, fromVertex, toVertex)

        assert nTriangleNPY==nTriangle
        assert nTriangleCPU==nTriangle

        if ops.local.cuda_enabled:
            nTriangleGPU = countTrianglesGPU(startEdge, fromVertex, toVertex)
            assert nTriangleGPU==nTriangle


#if __name__ == '__main__':
#    unittest.main()