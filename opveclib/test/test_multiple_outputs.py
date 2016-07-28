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
from ..operator import operator, evaluate
from ..expression import position_in, output_like
from ..local import cuda_enabled, clear_op_cache


@operator()
def multi_op(input0, input1, input2):
    """
    :param input0:
    :param input1:
    :param input2:
    :return: first output is the sum of first two inputs
        second output is the sum of the first two multiplied by the third
        third output is sum of all three inputs
    """
    pos = position_in(input0.shape)
    output0 = output_like(input0)
    output1 = output_like(input0)
    output2 = output_like(input0)

    a = input0[pos]
    b = input1[pos]
    c = input2[pos]
    d = a + b
    output0[pos] = d
    output1[pos] = d*c
    output2[pos] = d+c

    return output0, output1, output2


class TestMultipleOutputs(unittest.TestCase):
    clear_op_cache()

    def test(self):
        a = np.random.random(5)
        b = np.random.random(5)
        c = np.random.random(5)
        op = multi_op(a, b, c)
        op_c = evaluate(op, target_language='cpp')

        assert np.allclose(op_c[0], a+b)
        assert np.allclose(op_c[1], (a+b)*c)
        assert np.allclose(op_c[2], a+b+c)

        if cuda_enabled:
            op_cuda = evaluate(op, target_language='cuda')
            assert np.allclose(op_cuda[0], a+b)
            assert np.allclose(op_cuda[1], (a+b)*c)
            assert np.allclose(op_cuda[2], a+b+c)
