# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

from __future__ import print_function
import unittest
import numpy as np
from sys import _getframe
from ..operator import operator, evaluate
from ..expression import position_in, output
from ..local import cuda_enabled, clear_op_cache

@operator()
def split(in0):
    assert(in0.shape[0]==4)
    nCol = in0.shape[1]
    iCol = position_in(nCol)
    row0 = output(nCol, in0.dtype)
    row1 = output(nCol, in0.dtype)
    row2 = output(nCol, in0.dtype)
    row3 = output(nCol, in0.dtype)
    row0[iCol] = in0[0, iCol]
    row1[iCol] = in0[1, iCol]
    row2[iCol] = in0[2, iCol]
    row3[iCol] = in0[3, iCol]
    return row0, row1, row2, row3

@operator()
def add(in0, in1):
    assert(in0.shape==in1.shape)
    n = in0.shape[0]
    i = position_in(n)
    sumVal = output(n, in0.dtype)
    sumVal[i] = in0[i] + in1[i]
    return sumVal

@operator()
def mul(in0, in1):
    assert(in0.shape==in1.shape)
    n = in0.shape[0]
    i = position_in(n)
    prodVal = output(n, in0.dtype)
    prodVal[i] = in0[i] * in1[i]
    return prodVal

@operator()
def concatenate(op0, op1, op2):
    assert(op0.shape==op1.shape)
    assert(op1.shape==op2.shape)
    nCol = op0.shape[0]
    iCol = position_in(nCol)
    merged = output([3,nCol],op0.dtype)
    merged[0, iCol] = op0[iCol]
    merged[1, iCol] = op1[iCol]
    merged[2, iCol] = op2[iCol]
    return merged

@operator()
def fused(in0):
    assert(in0.shape[0]==4)
    nCol = in0.shape[1]
    iCol = position_in(nCol)
    merged = output([3,nCol],in0.dtype)
    t0 = in0[0,iCol] + in0[1,iCol]
    t1 = in0[2,iCol] + in0[3,iCol]
    merged[0,iCol] = t0
    merged[1,iCol] = t0*t1
    merged[2,iCol] = t1
    return merged

class TestOperator(unittest.TestCase):
    def testOperatorDag(self):
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        nRow = 4
        nCol = 5
        in0 = np.random.random([nRow,nCol])
        row0, row1, row2, row3 = split(in0)
        sum0 = add(row0, row1)
        sum1 = add(row2, row3)
        prod0 = mul(sum0, sum1)
        out0 = concatenate(sum0,prod0,sum1)
        out1 = fused(in0)

        if cuda_enabled:
            out0Eval = evaluate(out0, target_language='cuda')
            outMerge0Eval = evaluate(out1, target_language='cuda')
            np.allclose(out0Eval, outMerge0Eval)

if __name__ == '__main__':
    clear_op_cache()
    unittest.main()