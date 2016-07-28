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
import tensorflow as tf
from ..operator import operator, evaluate, as_tensorflow
from ..expression import position_in, output_like
from ..local import clear_op_cache
from .math import add, sub, mul, div, neg


class TestMath(unittest.TestCase):
    clear_op_cache()

    def test_binary_infix(self):
        """
        Make sure all binary infix operators yield the same result as numpy
        """
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

    def test_cwise_binary_grad(self):
        """
        Ensure that all component-wise binary functions in the math op library yield an identical gradient to tensorflow
        """
        with tf.Session() as s:
            lhs_np = np.random.random(100)
            rhs_np = lhs_np + np.random.randint(-1, 2, size=100)
            grad_above = tf.constant(np.random.random(100))

            lhs = tf.constant(lhs_np)
            rhs = tf.constant(rhs_np)

            def test_grad(fcn, tf_fcn):
                ovl_l = fcn(lhs, rhs)
                ovl_r = fcn(rhs, lhs)
                ovl_l, ovl_r = as_tensorflow([ovl_l, ovl_r])
                tf_l = tf_fcn(lhs, rhs)
                tf_r = tf_fcn(rhs, lhs)

                tf_grad_l, tf_grad_r = tf.gradients(tf_l, [lhs, rhs], grad_above)
                ovl_grad_l, ovl_grad_r = tf.gradients(ovl_l, [lhs, rhs], grad_above)

                ovl_l, ovl_r, tf_l, tf_r, tf_grad_l, tf_grad_r, ovl_grad_l, ovl_grad_r = s.run([
                    ovl_l, ovl_r, tf_l, tf_r, tf_grad_l, tf_grad_r, ovl_grad_l, ovl_grad_r])

                assert np.all(np.equal(ovl_l, tf_l))
                assert np.all(np.equal(ovl_r, tf_r))

                assert np.all(np.equal(tf_grad_l, ovl_grad_l))
                assert np.all(np.equal(tf_grad_r, ovl_grad_r))

            test_grad(lambda x, y: add(x, y), lambda x, y: tf.add(x, y))
            test_grad(lambda x, y: sub(x, y), lambda x, y: tf.sub(x, y))
            test_grad(lambda x, y: mul(x, y), lambda x, y: tf.mul(x, y))
            test_grad(lambda x, y: div(x, y), lambda x, y: tf.div(x, y))

    def test_cwise_unary_grad(self):
        """
        Ensure that all component-wise unary functions in the math op library yield an identical gradient to tensorflow
        """
        with tf.Session() as s:
            arg_np = np.random.random(100)
            grad_above = tf.constant(np.random.random(100))

            arg = tf.constant(arg_np)

            def test_grad(fcn, tf_fcn):
                ovl_out = as_tensorflow(fcn(arg))
                tf_out = tf_fcn(arg)

                ovl_grad = tf.gradients(ovl_out, arg, grad_above)[0]
                tf_grad = tf.gradients(tf_out, arg, grad_above)[0]
                ovl_out, tf_out, ovl_grad, tf_grad = s.run([ovl_out, tf_out, ovl_grad, tf_grad])

                assert np.all(np.equal(ovl_out, tf_out))
                assert np.all(np.equal(ovl_grad, tf_grad))

            test_grad(lambda x: neg(x), lambda x: tf.neg(x))
