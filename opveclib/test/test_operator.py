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
from ..operator import operator, _build_op_dag, evaluate
from ..expression import position_in, output_like, variable
from ..local import clear_op_cache


def index_by_id(items, item):
    item_id = id(item)
    for n, elem in enumerate(items):
        if id(elem) == item_id:
            return n

    raise ValueError('Item not found')


class TestOperator(unittest.TestCase):
    clear_op_cache()

    def test_dag_builder_simple(self):
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
        operators = op_dag.operators

        # assert correct dag structure
        first_index = index_by_id(operators, a1.parent)
        assert dag.references[first_index].input_refs[0].is_leaf is True
        assert inputs[dag.references[first_index].input_refs[0].dag_input_index] is in0
        assert dag.references[first_index].input_refs[1].is_leaf is True
        assert inputs[dag.references[first_index].input_refs[1].dag_input_index] is in1

        second_index = index_by_id(operators, a2.parent)
        assert second_index > first_index
        assert dag.references[second_index].input_refs[0].is_leaf is False
        assert dag.references[second_index].input_refs[0].op_index == first_index
        assert dag.references[second_index].input_refs[0].op_output_index == 0
        assert dag.references[second_index].input_refs[1].op_index == first_index
        assert dag.references[second_index].input_refs[1].op_output_index == 1

        # assert correct evaluation
        a1, b1, a2, b2 = evaluate([a1, b1, a2, b2])
        assert np.alltrue(a1 == in0+1)
        assert np.alltrue(b1 == in1+1)
        assert np.alltrue(a2 == in0+2)
        assert np.alltrue(b2 == in1+2)

    def test_dag_builder_complex(self):

        @operator()
        def add_one(x, y):
            assert x.shape == y.shape

            aa = output_like(x)
            bb = output_like(y)

            pos = position_in(aa.shape)

            aa[pos] = x[pos] + 1
            bb[pos] = y[pos] + 1

            return aa, bb

        in0 = np.random.random(10)
        in1 = np.random.random(10)
        a, b = add_one(in0, in1)
        c, d = add_one(a, in1)
        e, f = add_one(in1, b)
        g, h = add_one(f, a)
        in2 = np.random.random(10)
        i, j = add_one(e, in2)

        op_dag = _build_op_dag(i, h)
        operators = op_dag.operators
        dag = op_dag.proto_dag
        inputs = op_dag.inputs
        # neither i nor h are dependent on c or d, so their parent op should not be in op_dag
        try:
            index_by_id(operators, c.parent)
        except ValueError:
            pass
        else:
            raise AssertionError

        # assert correct dag structure for output a's parent
        first_index = index_by_id(operators, a.parent)
        assert dag.references[first_index].input_refs[0].is_leaf is True
        assert inputs[dag.references[first_index].input_refs[0].dag_input_index] is in0
        assert dag.references[first_index].input_refs[1].is_leaf is True
        assert inputs[dag.references[first_index].input_refs[1].dag_input_index] is in1

        # assert correct dag structure for output e's parent
        second_index = index_by_id(operators, e.parent)
        assert second_index > first_index
        assert dag.references[second_index].input_refs[0].is_leaf is True
        assert inputs[dag.references[second_index].input_refs[0].dag_input_index] is in1
        assert dag.references[second_index].input_refs[1].is_leaf is False
        assert dag.references[second_index].input_refs[1].op_index == first_index
        assert dag.references[second_index].input_refs[1].op_output_index == 1

        # assert correct dag structure for output g's parent
        third_index = index_by_id(operators, g.parent)
        assert third_index > second_index
        assert dag.references[third_index].input_refs[0].is_leaf is False
        assert dag.references[third_index].input_refs[0].op_index == second_index
        assert dag.references[third_index].input_refs[0].op_output_index == 1

        assert dag.references[third_index].input_refs[1].is_leaf is False
        assert dag.references[third_index].input_refs[1].op_index == first_index
        assert dag.references[third_index].input_refs[1].op_output_index == 0

        # assert correct dag structure for output i's parent
        fourth_index = index_by_id(operators, i.parent)
        assert fourth_index > second_index
        assert dag.references[fourth_index].input_refs[0].is_leaf is False
        assert dag.references[fourth_index].input_refs[0].op_index == second_index
        assert dag.references[fourth_index].input_refs[0].op_output_index == 0

        assert dag.references[fourth_index].input_refs[1].is_leaf is True
        assert inputs[dag.references[fourth_index].input_refs[1].dag_input_index] is in2

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

    def test_reassign(self):

        try:
            @operator()
            def bad(data_in):
                out = output_like(data_in)
                pos = position_in(data_in)

                a = variable(0, dtype=data_in.dtype)
                a = 2

                out[pos] = a
                return out
        except SyntaxError:
            pass
        else:
            raise AssertionError

        try:
            @operator()
            def bad(data_in):
                out = output_like(data_in)
                pos = position_in(data_in)

                a = variable(0, dtype=data_in.dtype)
                a = a + data_in[pos]

                out[pos] = a
                return out
        except SyntaxError:
            pass
        else:
            raise AssertionError
