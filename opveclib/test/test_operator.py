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
from ..operator import operator, _build_op_dag
from ..expression import position_in, output_like


#TODO: merge with test_output_reorder
#TODO: expand tests to make sure intelligible errors are thrown for bad syntax

@operator()
def add_one(x, y):
    assert x.shape == y.shape

    a = output_like(x)
    b = output_like(y)

    pos = position_in(a.shape)

    a[pos] = x[pos] + 1
    b[pos] = y[pos] + 1

    return a, b


class TestOperator(unittest.TestCase):
    def test_dag_builder(self):
        in1 = np.random.random(10)
        in2 = np.random.random(10)

        a1, b1 = add_one(in1, in2)

        a2, b2 = add_one(a1, b1)

        _build_op_dag(a2, b2)
        # TODO: test that _build_op_dag is getting the correct dag structure, and test some more complicated dags
        pass


if __name__ == '__main__':
    unittest.main()
