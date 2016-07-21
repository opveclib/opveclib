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
from ..expression import output, position_in, variable, uint32, float64, arange
from ..local import cuda_enabled, clear_op_cache


@operator()
def dot(input0, input1):
    num_vectors = input0.shape[0]

    dot_out = output(num_vectors, input0.dtype)

    pos = position_in(num_vectors)[0]

    accum = variable(0.0, float64)
    num_elements = variable(input0.shape[1], uint32)
    for i in arange(num_elements):
        accum <<= accum + input0[pos, i]*input1[pos, i]

    dot_out[pos] = accum

    return dot_out


class TestDot(unittest.TestCase):
    clear_op_cache()

    def test(self):
        rng = np.random.RandomState(1)
        x = rng.uniform(-1, 1, (100, 10))
        y = rng.uniform(-1, 1, (100, 10))
        op = dot(x, y)

        op_np = np.sum(x*y, axis=1)
        op_c = evaluate(op, target_language='cpp')
        assert np.allclose(op_c, op_np)

        if cuda_enabled:
            op_cuda = evaluate(op, target_language='cuda')
            assert np.allclose(op_cuda, op_np)
