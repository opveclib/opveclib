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
from ..operator import operator, _build_op_dag, evaluate
from ..expression import position_in, output_like
from ..local import clear_op_cache


class TestOperator(unittest.TestCase):
    def test_dag_builder(self):
        in0 = np.random.random(10)
        in1 = np.random.random(10)

        @operator()
        def add_one(x, y):
            assert x.shape == y.shape

            a = output_like(x)
            b = output_like(y)

            pos = position_in(a.shape)

            a[pos] = x[pos] + 1
            b[pos] = y[pos] + 1

            return a, b

        a1, b1 = add_one(in0, in1)
        a2, b2 = add_one(a1, b1)

        op_dag = _build_op_dag(a2, b2)
        dag = op_dag.proto_dag
        inputs = op_dag.inputs
        ops = op_dag.operators

        # assert correct dag structure
        first_index = ops.index(a1.parent)
        assert dag.references[first_index].input_refs[0].is_leaf is True
        assert inputs[dag.references[first_index].input_refs[0].dag_input_index] is in0
        assert dag.references[first_index].input_refs[1].is_leaf is True
        assert inputs[dag.references[first_index].input_refs[1].dag_input_index] is in1

        second_index = ops.index(a2.parent)
        assert second_index > first_index
        assert dag.references[second_index].input_refs[0].is_leaf is False
        assert dag.references[second_index].input_refs[0].op_index == first_index
        assert dag.references[second_index].input_refs[0].op_output_index == 0
        assert dag.references[second_index].input_refs[1].op_index == first_index
        assert dag.references[second_index].input_refs[1].op_output_index == 1
        assert inputs[dag.references[0].input_refs[1].dag_input_index] is in1

        # assert correct evaluation
        a1, b1, a2, b2 = evaluate([a1, b1, a2, b2])
        assert np.alltrue(a1 == in0+1)
        assert np.alltrue(b1 == in1+1)
        assert np.alltrue(a2 == in0+2)
        assert np.alltrue(b2 == in1+2)

        # a, b = add_one(in0, in1)
        # c, d = add_one(a, in1)
        # d, e = add_one(in1, b)
        # f, g = add_one(c, e)
        # h, i = add_one(g, d)

    # Operation should reorder the output io_index to be in the order with which they are returned, not the order
    # in which they are declared
    def test_return_reordering(self):
        @operator()
        def reorder(input0):
            pos = position_in(input0.shape)

            # declare out of order
            output3 = output_like(input0)
            output0 = output_like(input0)
            output2 = output_like(input0)
            output1 = output_like(input0)

            inp = input0[pos]

            output0[pos] = 2*inp
            output1[pos] = 3*inp
            output2[pos] = 4*inp
            output3[pos] = 5*inp

            # return in order
            return output0, output1, output2, output3

        a = np.random.random(5)
        op = reorder(a)
        o0, o1, o2, o3 = evaluate(op, target_language='cpp')

        assert np.alltrue(np.equal(o0, 2*a))
        assert np.alltrue(np.equal(o1, 3*a))
        assert np.alltrue(np.equal(o2, 4*a))
        assert np.alltrue(np.equal(o3, 5*a))

    # Operation should throw an error when user tries to return the wrong number of outputs or the wrong type of value
    def test_output_error(self):
        @operator()
        def too_few(input0):
            pos = position_in(input0.shape)

            # declare out of order
            output3 = output_like(input0)
            output0 = output_like(input0)
            output2 = output_like(input0)
            output1 = output_like(input0)

            input_value = input0[pos]

            output0[pos] = 2*input_value
            output1[pos] = 3*input_value
            output2[pos] = 4*input_value
            output3[pos] = 5*input_value

            # return in order
            return output0, output1

        a = np.random.random(5)
        try:
            too_few(a)
        except ValueError:
            pass
        else:
            assert False

        @operator()
        def no_outputs(input0):
            pos = position_in(input0.shape)

            input_value = input0[pos]
            output0 = output_like(input0)
            output0[pos] = 2*input_value

        try:
            no_outputs(a)
        except ValueError:
            pass
        else:
            assert False

        @operator()
        def bad_return(input0):
            pos = position_in(input0.shape)

            input_value = input0[pos]
            output0 = output_like(input0)
            output0[pos] = 2*input_value

            return input_value

        try:
            bad_return(a)
        except TypeError:
            pass
        else:
            assert False

if __name__ == '__main__':
    clear_op_cache()
    unittest.main()
