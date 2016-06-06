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
from ..operator import Operator
from ..expression import output, position_in, variable, uint32, float64, arange
from ..local import cuda_enabled


class Dot(Operator):
    def op(self, input0, input1):
        num_vectors = input0.shape[0]

        dot = output(num_vectors, input0.dtype)

        pos = position_in(num_vectors)[0]

        accum = variable(0.0, float64)
        num_elements = variable(input0.shape[1], uint32)
        for i in arange(num_elements):
            accum <<= accum + input0[pos, i]*input1[pos, i]

        dot[pos] = accum

        return dot


class TestDot(unittest.TestCase):
    def test(self):
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        rng = np.random.RandomState(1)
        x = rng.uniform(-1, 1, (100, 10))
        y = rng.uniform(-1, 1, (100, 10))
        op = Dot(x, y, clear_cache=True)

        op_np = np.sum(x*y, axis=1)
        op_c = op.evaluate_c()
        assert np.allclose(op_c, op_np)

        if cuda_enabled:
            op_cuda = op.evaluate_cuda()
            assert np.allclose(op_cuda, op_np)

if __name__ == '__main__':
    unittest.main()
