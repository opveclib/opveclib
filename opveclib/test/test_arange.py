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
from ..expression import output, position_in, arange, cast
from ..local import cuda_enabled


class FillRange(Operator):
    """
    Fill two output arrays, one increasing to the right, one decreasing to the left
    """
    def op(self, input0):
        assert len(input0.shape) == 1
        pos = position_in([1])[0]

        num_elements = input0.shape[0]
        out_right = output(input0.shape, input0.dtype)
        out_left = output(input0.shape, input0.dtype)
        for i in arange(num_elements):
            out_right[i] = input0[i]*cast(i, input0.dtype)

        for i in arange(num_elements-1, -1, -1):
            out_left[i] = input0[i]*cast(i, input0.dtype)

        return out_right, out_left


class TestArange(unittest.TestCase):
    def test(self):
        print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
        rng = np.random.RandomState(1)
        num_elements = 10
        x = rng.uniform(-1, 1, num_elements)

        op = FillRange(x, clear_cache=True)

        np_ref = x * np.arange(num_elements)
        right_c, left_c = op.evaluate_c()
        assert np.all(right_c == np_ref)
        assert np.all(left_c == np_ref)

        if cuda_enabled:
            right_cuda, left_cuda = op.evaluate_cuda()
            assert np.all(right_cuda == np_ref)
            assert np.all(left_cuda == np_ref)

if __name__ == '__main__':
    unittest.main()
