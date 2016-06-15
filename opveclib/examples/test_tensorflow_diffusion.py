# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.


import numpy as np
import unittest
from sys import _getframe
from .diffusion import diffusion2DGPU, diffusion2DNp
import opveclib as ops

class TestDiffusion2D(unittest.TestCase):
    """
    Test cases for diffusion in 2D.
    """
    def test(self):
        """
        Downloads an image
        These test cases use the numpy reference implementation of the operator to compare against the opveclib
        implementation.
        """
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)

        # CUDA is required to run these.
        if ops.local.cuda_enabled:
            for nX in [15, 20]:
                for nY in [15, 50]:
                    print("Test case nX = %d and nY = %d." % (nX, nY)) # Print parameters of the test case.
                    rng     = np.random.RandomState(1)
                    imageIn = rng.uniform(0, 1, [nY, nX])

                    imageGPU    = diffusion2DGPU(imageIn, dt=5, l=3.5/255, s=3, nIter=3)
                    imageNPY    = diffusion2DNp(imageIn, dt=5, l=3.5/255, s=3, nIter=3)

                    assert np.allclose(imageGPU, imageNPY)

if __name__ == '__main__':
    unittest.main()