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
from .diffusion import AddBoundaryOp, addBoundaryNp, DelBoundaryOp, delBoundaryNp, CopyBoundaryOp, copyBoundaryNp
from .diffusion import DiffusionGradient2DOp, diffusionGradient2DNp
from .diffusion import SolveDiagCol2DOp, SolveDiagRow2DOp, solveDiagCol2DNp, solveDiagRow2DNp
import opveclib as ops


class TestAddBoundary(unittest.TestCase):
    """
    Test cases for the add boundary operator.
    """
    def test(self):
        """
        Compares the numpy reference implementation to the opveclib implementation for the add operator.
        """
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        for nX in [2, 3, 10]:
            for nY in [2, 3, 5]:
                print("Test case nX = %d and nY = %d." % (nX, nY)) # Print parameters of the test case.
                rng     = np.random.RandomState(1)
                dataIn  = rng.uniform(0, 255, [nY, nX])
                op      = AddBoundaryOp(dataIn, clear_cache=True)

                dataCPU = op.evaluate_c()
                dataNPY = addBoundaryNp(dataIn)

                assert np.allclose(dataCPU, dataNPY)

                if ops.local.cuda_enabled:
                    dataGPU = op.evaluate_cuda()
                    assert np.allclose(dataGPU, dataNPY)



class TestDelBoundary(unittest.TestCase):
    """
    Test cases for deletion boundary operator.
    """
    def test(self):
        """
        Compare the numpy implementation to the opveclib implementation for the deletion boundary operator.
        """
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        for nX in [3, 4, 10]:
            for nY in [3, 4, 5]:
                print("Test case nX = %d and nY = %d." % (nX, nY)) # Print parameters of the test case.

                rng     = np.random.RandomState(1)
                dataIn  = rng.uniform(0, 255, [nY, nX])
                op      = DelBoundaryOp(dataIn, clear_cache=True)
                dataCPU = op.evaluate_c()
                dataNPY = delBoundaryNp(dataIn)
                assert np.allclose(dataCPU, dataNPY)

                if ops.local.cuda_enabled:
                    dataGPU = op.evaluate_cuda()
                    assert np.allclose(dataGPU, dataNPY)



class TestCopyBoundary(unittest.TestCase):
    """
    Test cases for the copy boundary operator.
    """
    def test(self):
        """
        Compare the numpy implementation with the opveclib implementation for the copy boundary operator.
        """
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        for nX in [2, 3, 10]:
            for nY in [2, 3, 5]:
                print("Test case nX = %d and nY = %d." % (nX, nY)) # Print parameters of the test case.

                rng     = np.random.RandomState(1)
                dataIn  = rng.uniform(0, 255, [nY, nX])
                op      = CopyBoundaryOp(dataIn, clear_cache=True)
                dataCPU = op.evaluate_c()
                dataNPY = copyBoundaryNp(dataIn)
                assert np.allclose(dataCPU, dataNPY)

                if ops.local.cuda_enabled:
                    dataGPU = op.evaluate_cuda()
                    assert np.allclose(dataGPU, dataNPY)


class TestDiffusionGradient(unittest.TestCase):
    """
    Test cases for the diffusion gradient operator.
    """
    def test(self):
        """
        Compare the numpy implementation with the opveclib implementation for the diffusion gradient operator.
        """
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        for nX in [3, 4, 10]:
            for nY in [3, 4, 5]:
                rng = np.random.RandomState(1)
                image = rng.uniform(0, 255, [nY, nX])
                l = 3.5/255
                l2 = l*l
                op = DiffusionGradient2DOp(image, l2=l2, clear_cache=True)
                gradRowPlusCPU, gradRowMinusCPU, gradColPlusCPU, gradColMinusCPU = op.evaluate_c()
                gradRowPlusNPY, gradRowMinusNPY, gradColPlusNPY, gradColMinusNPY = diffusionGradient2DNp(image, l2)

                assert np.allclose(gradRowPlusCPU, gradRowPlusNPY)
                assert np.allclose(gradRowMinusCPU, gradRowMinusNPY)
                assert np.allclose(gradColPlusCPU, gradColPlusNPY)
                assert np.allclose(gradColMinusCPU, gradColMinusNPY)

                if ops.local.cuda_enabled:
                    gradRowPlusGPU, gradRowMinusGPU, gradColPlusGPU, gradColMinusGPU = op.evaluate_cuda()
                    assert np.allclose(gradRowPlusGPU, gradRowPlusNPY)
                    assert np.allclose(gradRowMinusGPU, gradRowMinusNPY)
                    assert np.allclose(gradColPlusGPU, gradColPlusNPY)
                    assert np.allclose(gradColMinusGPU, gradColMinusNPY)


class TestSolveDiag2DOp(unittest.TestCase):
    """
    Test cases for the solve diag 2D operator.
    """
    def test(self):
        """
        Compare the numpy implementation with the opveclib implementation for the solve diag 2D operator for columns and
        for rows.
        """
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        for nY in [1, 2, 3, 7]:
            for nX in [1, 2, 3, 4]:
                print("Test case nX = %d and nY = %d." % (nX, nY))
                rng = np.random.RandomState(1)

                alpha   = rng.uniform(0, 1, [nY, nX])
                beta    = rng.uniform(0, 1, [nY, nX])
                gamma   = rng.uniform(0, 1, [nY, nX])
                y       = rng.uniform(0, 1, [nY, nX])

                opRow   = SolveDiagRow2DOp(alpha, beta, gamma, y, clear_cache=True)
                xRowCPU = opRow.evaluate_c()
                xRowNPY = solveDiagRow2DNp(alpha, beta, gamma, y)

                opCol   = SolveDiagCol2DOp(alpha, beta, gamma, y, clear_cache=True)
                xColCPU = opCol.evaluate_c()
                xColNPY = solveDiagCol2DNp(alpha, beta, gamma, y)

                assert np.allclose(xColCPU, xColNPY)
                assert np.allclose(xRowCPU, xRowNPY)

                if ops.local.cuda_enabled:
                    xRowGPU = opRow.evaluate_cuda()
                    xColGPU = opCol.evaluate_cuda()
                    assert np.allclose(xColGPU, xColNPY)
                    assert np.allclose(xRowGPU, xRowNPY)


if __name__ == '__main__':
    unittest.main()