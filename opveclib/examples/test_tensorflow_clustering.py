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
from clustering import createClusterData, initialClusterCenters, kMeansTF, kMeansGPU, compareClusterCenters
from sys import _getframe
import numpy as np
import opveclib as ops

class TestKMeans(unittest.TestCase):
    """
    Test cases for kMeans clustering.
    """
    def test(self):
        """
        This test case generates random data with eight cluster centers and 100 data points per cluster center. Each
        datum has three components.
        """
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        np.random.seed(3)

        data, clusterCenterGt = createClusterData()
        clusterCenter = initialClusterCenters()

        # CUDA is not required to run these test cases.
        # TF will automatically run on the CPU version if that is all that is available
        clusterCenterOVL    = kMeansGPU(data, clusterCenter, nMaxIter=500, th=1e-4)
        clusterCenterTF     = kMeansTF(data, clusterCenter, nMaxIter=500, th=1e-4)

        assert compareClusterCenters(clusterCenterGt, clusterCenterOVL, rtol=0.1, atol=0.1)
        assert compareClusterCenters(clusterCenterGt, clusterCenterTF, rtol=0.1, atol=0.1)