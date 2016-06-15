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
import opveclib as ops
from sys import _getframe
from .diffusion import Filter2DOp, filter2DNp


class TestFilter2D(unittest.TestCase):
    """
    Test cases for filtering operator.
    """
    def test(self):
        """
        Compare the numpy implementation with the opveclib implementation for varying image size and varying kernel
        size.
        """
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        for nY in [5, 8, 15]:
            for nX in [5, 9, 16]:
                print("Image size nX = %d and nY = %d." % (nX, nY)) # Print parameters of the test case.

                for mY in [2, 3]:
                    for mX in [2, 3]:
                        print("  Kernel size mX = %d and mY = %d." % (mX, mY)) # Print parameters of the test case.

                        rng         = np.random.RandomState(1)
                        dataIn      = rng.uniform(0, 255, [nY, nX])
                        kernelIn    = rng.uniform(0, 255, [mY, mX])
                        op          = Filter2DOp(dataIn, kernelIn, clear_cache=True)
                        dataCPU     = op.evaluate_c()
                        dataNPY     = filter2DNp(dataIn, kernelIn)

                        assert np.allclose(dataCPU, dataNPY)

                        if ops.local.cuda_enabled:
                            dataGPU = op.evaluate_cuda()
                            assert np.allclose(dataGPU, dataNPY)

if __name__ == '__main__':
    unittest.main()