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
from ..expression import position_in, output_like, if_, elif_, else_
from ..local import cuda_enabled, clear_op_cache


@operator()
def clip(arg, threshold1=None, threshold2=None):
    pos = position_in(arg.shape)

    clipped = output_like(arg)
    x = arg[pos]

    with if_(x < threshold1):
        clipped[pos] = threshold1

    with elif_(x > threshold2):
        clipped[pos] = threshold2

    with else_():
        clipped[pos] = x

    return clipped


class TestClip(unittest.TestCase):
    clear_op_cache()

    def test(self):
        a = np.random.random(1000)

        op = clip(a, threshold1=0.1, threshold2=0.9)
        op_c = evaluate(op, target_language='cpp')
        op_np = np.clip(a, 0.1, 0.9)
        assert np.all(np.equal(op_c, op_np))

        if cuda_enabled:
            op_cuda = evaluate(op, target_language='cuda')
            assert np.all(np.equal(op_cuda, op_np))
