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
from ..expression import position_in, output, min_value, max_value, epsilon
from ..local import cuda_enabled, clear_op_cache


@operator()
def set_limits(arg):
    pos = position_in(arg.shape)
    limits = output([arg.shape[0], 3], arg.dtype)
    t = arg.dtype
    limits[pos,0] = min_value(t)
    limits[pos,1] = max_value(t)
    limits[pos,2] = epsilon(t)

    return limits


class TestLimits(unittest.TestCase):
    clear_op_cache()

    def test(self):
        a = np.array([0])
        types = [np.float32, np.float64]
        for t in types:
            op = set_limits(a.astype(t))
            op_c = evaluate(op, target_language='cpp')
            op_np = np.array([[np.finfo(t).tiny, np.finfo(t).max, np.finfo(t).eps]])
            assert np.all(np.equal(op_c, op_np))

            if cuda_enabled:
                op_cuda = evaluate(op, target_language='cuda')
                assert np.all(np.equal(op_cuda, op_np))
