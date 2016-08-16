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
from .math import add, sub, mul, div, neg, tanh, sigmoid, split, concat


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

                assert np.allclose(ovl_out, tf_out)
                assert np.allclose(ovl_grad, tf_grad)

            test_grad(lambda x: neg(x), lambda x: tf.neg(x))
            test_grad(lambda x: tanh(x), lambda x: tf.tanh(x))
            test_grad(lambda x: sigmoid(x), lambda x: tf.sigmoid(x))

    def test_concat(self):
        with tf.Session() as s:
            num_concat = 5
            args_1d = []
            args_2d = []
            args_3d = []
            args_4d = []
            for n in range(num_concat):
                args_1d.append(tf.constant(np.random.random(10/num_concat).reshape((10/num_concat))))
                args_2d.append(tf.constant(np.random.random(100/num_concat).reshape((10, 10/num_concat))))
                args_3d.append(tf.constant(np.random.random(1000/num_concat).reshape((10, 10, 10/num_concat))))
                args_4d.append(tf.constant(np.random.random(10000/num_concat).reshape((10, 10, 10, 10/num_concat))))

            tf_1d = tf.concat(0, args_1d)
            tf_2d = tf.concat(1, args_2d)
            tf_3d = tf.concat(2, args_3d)
            tf_4d = tf.concat(3, args_4d)

            ovl_1d = as_tensorflow(concat(*args_1d, concat_dim=0))
            ovl_2d = as_tensorflow(concat(*args_2d, concat_dim=1))
            ovl_3d = as_tensorflow(concat(*args_3d, concat_dim=2))
            ovl_4d = as_tensorflow(concat(*args_4d, concat_dim=3))

            assert np.all(np.equal(*s.run([tf_1d, ovl_1d])))
            assert np.all(np.equal(*s.run([tf_2d, ovl_2d])))
            assert np.all(np.equal(*s.run([tf_3d, ovl_3d])))
            assert np.all(np.equal(*s.run([tf_4d, ovl_4d])))

            grad_above_1d = tf.constant(np.random.random(10))
            grad_above_2d = tf.constant(np.random.random(100).reshape((10, 10)))
            grad_above_3d = tf.constant(np.random.random(1000).reshape((10, 10, 10)))
            grad_above_4d = tf.constant(np.random.random(10000).reshape((10, 10, 10, 10)))

            tf_grads_1d = tf.gradients(tf_1d, args_1d, grad_above_1d)
            tf_grads_2d = tf.gradients(tf_2d, args_2d, grad_above_2d)
            tf_grads_3d = tf.gradients(tf_3d, args_3d, grad_above_3d)
            tf_grads_4d = tf.gradients(tf_4d, args_4d, grad_above_4d)

            ovl_grads_1d = tf.gradients(ovl_1d, args_1d, grad_above_1d)
            ovl_grads_2d = tf.gradients(ovl_2d, args_2d, grad_above_2d)
            ovl_grads_3d = tf.gradients(ovl_3d, args_3d, grad_above_3d)
            ovl_grads_4d = tf.gradients(ovl_4d, args_4d, grad_above_4d)

            for grad_index in range(num_concat):
                assert np.all(np.equal(*s.run([tf_grads_1d[grad_index], ovl_grads_1d[grad_index]])))
                assert np.all(np.equal(*s.run([tf_grads_2d[grad_index], ovl_grads_2d[grad_index]])))
                assert np.all(np.equal(*s.run([tf_grads_3d[grad_index], ovl_grads_3d[grad_index]])))
                assert np.all(np.equal(*s.run([tf_grads_4d[grad_index], ovl_grads_4d[grad_index]])))

    def test_split(self):
        with tf.Session() as s:
            arg_1d = tf.constant(np.random.random(10))
            arg_2d = tf.constant(np.random.random(100).reshape((10, 10)))
            arg_3d = tf.constant(np.random.random(1000).reshape((10, 10, 10)))
            arg_4d = tf.constant(np.random.random(10000).reshape((10, 10, 10, 10)))

            num_split = 5
            ovl_1d = as_tensorflow(split(arg_1d, split_dim=0, num_split=num_split))
            ovl_2d = as_tensorflow(split(arg_2d, split_dim=1, num_split=num_split))
            ovl_3d = as_tensorflow(split(arg_3d, split_dim=2, num_split=num_split))
            ovl_4d = as_tensorflow(split(arg_4d, split_dim=3, num_split=num_split))

            tf_1d = tf.split(0, num_split, arg_1d)
            tf_2d = tf.split(1, num_split, arg_2d)
            tf_3d = tf.split(2, num_split, arg_3d)
            tf_4d = tf.split(3, num_split, arg_4d)

            for n in range(num_split):
                assert np.all(np.equal(*s.run([ovl_1d[n], tf_1d[n]])))
                assert np.all(np.equal(*s.run([ovl_2d[n], tf_2d[n]])))
                assert np.all(np.equal(*s.run([ovl_3d[n], tf_3d[n]])))
                assert np.all(np.equal(*s.run([ovl_4d[n], tf_4d[n]])))

            grads_above_1d = []
            grads_above_2d = []
            grads_above_3d = []
            grads_above_4d = []
            for n in range(num_split):
                grads_above_1d.append(tf.constant(np.random.random(10/num_split).reshape((10/num_split))))
                grads_above_2d.append(tf.constant(np.random.random(100/num_split).reshape((10, 10/num_split))))
                grads_above_3d.append(tf.constant(np.random.random(1000/num_split).reshape((10, 10, 10/num_split))))
                grads_above_4d.append(tf.constant(np.random.random(10000/num_split).reshape((10, 10, 10, 10/num_split))))

            tf_grad_1d = tf.gradients(tf_1d, arg_1d, grads_above_1d)[0]
            ovl_grad_1d = tf.gradients(ovl_1d, arg_1d, grads_above_1d)[0]
            assert np.all(np.equal(*s.run([tf_grad_1d, ovl_grad_1d])))

            tf_grad_2d = tf.gradients(tf_2d, arg_2d, grads_above_2d)[0]
            ovl_grad_2d = tf.gradients(ovl_2d, arg_2d, grads_above_2d)[0]
            print(s.run([tf_grad_2d, ovl_grad_2d]))
            assert np.all(np.equal(*s.run([tf_grad_2d, ovl_grad_2d])))

            tf_grad_3d = tf.gradients(tf_3d, arg_3d, grads_above_3d)[0]
            ovl_grad_3d = tf.gradients(ovl_3d, arg_3d, grads_above_3d)[0]
            assert np.all(np.equal(*s.run([tf_grad_3d, ovl_grad_3d])))

            tf_grad_4d = tf.gradients(tf_4d, arg_4d, grads_above_4d)[0]
            ovl_grad_4d = tf.gradients(ovl_4d, arg_4d, grads_above_4d)[0]
            assert np.all(np.equal(*s.run([tf_grad_4d, ovl_grad_4d])))

    def test_lstm(self):

        batches = 200
        vec_len = 500
        forget = 0.0

        with tf.Session() as s:
            concat_arg = tf.constant(np.random.normal(size=batches*4*vec_len).reshape((batches, 4*vec_len)))
            c = tf.constant(np.random.normal(size=batches*vec_len).reshape((batches, vec_len)))

            i, j, f, o = tf.split(1, 4, concat_arg)

            new_c_tf = tf.mul(c,  tf.sigmoid(f + forget)) + tf.sigmoid(i) * tf.tanh(j)
            new_h_tf = tf.tanh(new_c_tf) * tf.sigmoid(o)

            i, j, f, o = split(concat_arg, split_dim=1, num_split=4)

            new_c = mul(c,  sigmoid(f)) + sigmoid(i) * tanh(j)
            new_h = tanh(new_c) * sigmoid(o)
            new_c_ovl, new_h_ovl = as_tensorflow([new_c, new_h])

            assert np.allclose(*s.run([new_c_tf, new_c_ovl]))
            assert np.allclose(*s.run([new_h_tf, new_h_ovl]))

            c_grad = tf.constant(np.random.normal(size=batches*vec_len).reshape((batches, vec_len)))
            h_grad = tf.constant(np.random.normal(size=batches*vec_len).reshape((batches, vec_len)))

            concat_grad_tf = tf.gradients([new_c_tf, new_h_tf], concat_arg, [c_grad, h_grad])[0]
            concat_grad_ovl = tf.gradients([new_c_ovl, new_h_ovl], concat_arg, [c_grad, h_grad])[0]

            c_grad_tf = tf.gradients([new_c_tf, new_h_tf], c, [c_grad, h_grad])[0]
            c_grad_ovl = tf.gradients([new_c_ovl, new_h_ovl], c, [c_grad, h_grad])[0]

            assert np.allclose(*s.run([concat_grad_tf, concat_grad_ovl]))
            assert np.allclose(*s.run([c_grad_tf, c_grad_ovl]))
