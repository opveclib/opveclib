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
from ..local import clear_op_cache


# def index_by_id(items, item):
#     item_id = id(item)
#     for n, elem in enumerate(items):
#         if id(elem) == item_id:
#             return n
#
#     raise ValueError('Item not found')


class TestMath(unittest.TestCase):
    clear_op_cache()

    def test_binary_infix(self):
        @operator()
        def no_op(x):
            out = output_like(x)
            pos = position_in(x.shape)

            out[pos] = x[pos]

            return out

        def test_infix(lhs, rhs):
            ovl0 = no_op(lhs)

            def test_np(fcn):
                ovl_l = fcn(ovl0, rhs)
                ovl_r = fcn(rhs, ovl0)
                ovl_l, ovl_r = evaluate([ovl_l, ovl_r])
                np_l = fcn(lhs, rhs)
                np_r = fcn(rhs, lhs)

                assert np.all(np.equal(ovl_l, np_l))
                assert np.all(np.equal(ovl_r, np_r))

            test_np(lambda x, y: x + y)
            test_np(lambda x, y: x - y)
            test_np(lambda x, y: x * y)
            test_np(lambda x, y: x / y)
            test_np(lambda x, y: x == y)
            test_np(lambda x, y: x != y)
            test_np(lambda x, y: x < y)
            test_np(lambda x, y: x <= y)
            test_np(lambda x, y: x > y)
            test_np(lambda x, y: x >= y)

            # OVL uses c-style fmod, not python style mod, so use numpy fmod function for test
            # see : http://docs.scipy.org/doc/numpy/reference/generated/numpy.fmod.html
            ovl_left, ovl_right = evaluate([ovl0 % rhs, rhs % ovl0])
            np_left = np.fmod(lhs, rhs)
            np_right = np.fmod(rhs, lhs)

            assert np.all(np.equal(ovl_left, np_left))
            assert np.all(np.equal(ovl_right, np_right))

            ovl_neg = evaluate([-ovl0])
            assert np.all(np.equal(ovl_neg, -lhs))

        np0 = np.random.random(100).reshape((10, 10))
        np1 = np0 + np.random.randint(-1, 2, size=100).reshape((10, 10))

        # test all binary infix operators
        test_infix(np0, np1)
        # test broadcasting from a scalar on rhs
        test_infix(np0, np.random.random(1))
        # test broadcasting from a scalar on lhs
        test_infix(np.random.random(1), np0)

        # attempting to negate an unsigned tensor should raise a type error
        uint = np.random.randint(0, 10, size=10).astype(np.uint8)
        try:
            -no_op(uint)
        except TypeError:
            pass
        else:
            raise AssertionError

    # def test_binary_infix_grad(self):
    #     @operator()
    #     def no_op(x):
    #         out = output_like(x)
    #         pos = position_in(x.shape)
    #
    #         out[pos] = x[pos]
    #
    #         return out
    #
    #
    #
    #
    #     with tf.Session() as s:
    #         lhs_np = np.random.random(100)
    #         rhs_np = lhs_np + np.random.randint(-1, 2, size=100)
    #         lhs = tf.constant(lhs_np)
    #         rhs = tf.constant(rhs_np)
    #         ovl_lhs = no_op(lhs)
    #
    #         def test_grad(fcn):
    #             ovl_l = fcn(ovl_lhs, rhs)
    #             ovl_r = fcn(rhs, ovl_lhs)
    #             ovl_l, ovl_r = as_tensorflow([ovl_l, ovl_r])
    #             tf_l = fcn(lhs, rhs)
    #             tf_r = fcn(rhs, lhs)
    #
    #             ovl_l, ovl_r, tf_l, tf_r = s.run([ovl_l, ovl_r, tf_l, tf_r])
    #
    #             assert np.all(np.equal(ovl_l, tf_l))
    #             assert np.all(np.equal(ovl_r, tf_r))
    #
    #         test_grad(lambda x, y: x + y)
