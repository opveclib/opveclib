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
    # clear_op_cache()

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
                args_2d.append(tf.constant(np.random.random(30/num_concat).reshape((3, 10/num_concat))))
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

    #
    # class LSTMP(Operator):
    #     def op(self, concat, c, forget_bias):
    #         batches = concat.shape[0]
    #         vec_len = concat.shape[1]/4
    #
    #         assert c.shape[0] == concat.shape[0]
    #         assert c.shape[1] == vec_len
    #         assert c.dtype == concat.dtype
    #
    #         pos = position_in([batches, vec_len])
    #         cur_batch = pos[0]
    #         cur_elem = pos[1]
    #
    #         i = concat[cur_batch, cur_elem]
    #         j = concat[cur_batch, cur_elem + vec_len]
    #         f = concat[cur_batch, cur_elem + 2*vec_len]
    #         o = concat[cur_batch, cur_elem + 3*vec_len]
    #         c_cur = c[cur_batch, cur_elem]
    #
    #         new_c = output_like(c)
    #         new_h = output_like(c)
    #
    #         new_c_cur = c_cur*sig(f + forget_bias) + sig(i) * tanh(j)
    #
    #         new_c[pos] = new_c_cur
    #         new_h[pos] = tanh(new_c_cur) * sig(o)
    #
    #         return new_c, new_h
    #
    #     def grad(self, concat, c, forget_bias, d_new_c, d_new_h):
    #         batches = concat.shape[0]
    #         vec_len = concat.shape[1]/4
    #
    #         assert c.shape[0] == concat.shape[0]
    #         assert c.shape[1] == vec_len
    #         assert c.dtype == concat.dtype
    #
    #         assert d_new_c.tensor_type == c.tensor_type
    #         assert d_new_h.tensor_type == c.tensor_type
    #
    #         pos = position_in([batches, vec_len])
    #         cur_batch = pos[0]
    #         cur_elem = pos[1]
    #
    #         i = concat[cur_batch, cur_elem]
    #         j = concat[cur_batch, cur_elem + vec_len]
    #         f = concat[cur_batch, cur_elem + 2*vec_len]
    #         o = concat[cur_batch, cur_elem + 3*vec_len]
    #         c_cur = c[cur_batch, cur_elem]
    #         new_c_cur = c_cur*sig(f + forget_bias) + sig(i) * tanh(j)
    #
    #         d_new_c_cur = d_new_c[cur_batch, cur_elem]
    #         d_new_h_cur = d_new_h[cur_batch, cur_elem]
    #
    #         d_concat = output_like(concat)
    #         d_c = output_like(c)
    #
    #         back_ch = d_new_c_cur + tanh_grad(new_c_cur)*sig(o)*d_new_h_cur
    #         d_i = tanh(j)*sig_grad(i)*back_ch
    #         d_j = sig(i)*tanh_grad(j)*back_ch
    #         d_f = c_cur*sig_grad(f+forget_bias)*back_ch
    #         d_c_cur = sig(f+forget_bias)*back_ch
    #         d_o = tanh(new_c_cur)*sig_grad(o)*d_new_h_cur
    #
    #         d_concat[cur_batch, cur_elem] = d_i
    #         d_concat[cur_batch, cur_elem+vec_len] = d_j
    #         d_concat[cur_batch, cur_elem+2*vec_len] = d_f
    #         d_concat[cur_batch, cur_elem+3*vec_len] = d_o
    #         d_c[pos] = d_c_cur
    #
    #         return d_concat, d_c
    #
    #
    # class LSTMP_jacobian(Operator):
    #     def op(self, concat, c, forget_bias, d_concat, d_c):
    #         batches = concat.shape[0]
    #         vec_len = concat.shape[1]/4
    #
    #         assert c.shape[0] == concat.shape[0]
    #         assert c.shape[1] == vec_len
    #         assert c.dtype == concat.dtype
    #
    #         assert d_concat.tensor_type == concat.tensor_type
    #         assert d_c.tensor_type == c.tensor_type
    #
    #         pos = position_in([batches, vec_len])
    #         cur_batch = pos[0]
    #         cur_elem = pos[1]
    #
    #         i = concat[cur_batch, cur_elem]
    #         j = concat[cur_batch, cur_elem + vec_len]
    #         f = concat[cur_batch, cur_elem + 2*vec_len]
    #         o = concat[cur_batch, cur_elem + 3*vec_len]
    #         c_cur = c[cur_batch, cur_elem]
    #         new_c_cur = c_cur*sig(f + forget_bias) + sig(i) * tanh(j)
    #
    #         d_new_c = output_like(c)
    #         d_new_h = output_like(c)
    #
    #         d_i = d_concat[cur_batch, cur_elem]
    #         d_j = d_concat[cur_batch, cur_elem + vec_len]
    #         d_f = d_concat[cur_batch, cur_elem + 2*vec_len]
    #         d_o = d_concat[cur_batch, cur_elem + 3*vec_len]
    #         d_c_cur = d_c[cur_batch, cur_elem]
    #
    #         d_new_c_cur = c_cur*sig_grad(f+forget_bias)*d_f + \
    #             sig(f+forget_bias)*d_c_cur + \
    #             tanh(j)*sig_grad(i)*d_i + \
    #             sig(i)*tanh_grad(j)*d_j
    #
    #         d_new_h_cur = sig(o)*tanh_grad(new_c_cur)*d_new_c_cur + \
    #             tanh(new_c_cur)*sig_grad(o)*d_o
    #
    #         d_new_c[pos] = d_new_c_cur
    #         d_new_h[pos] = d_new_h_cur
    #
    #         return d_new_c, d_new_h
    #
    #
    # class LSTMP_jacobian_adjoint(Operator):
    #     def op(self, concat, c, forget_bias, d_new_c, d_new_h):
    #         batches = concat.shape[0]
    #         vec_len = concat.shape[1]/4
    #
    #         assert c.shape[0] == concat.shape[0]
    #         assert c.shape[1] == vec_len
    #         assert c.dtype == concat.dtype
    #
    #         assert d_new_c.tensor_type == c.tensor_type
    #         assert d_new_h.tensor_type == c.tensor_type
    #
    #         pos = position_in([batches, vec_len])
    #         cur_batch = pos[0]
    #         cur_elem = pos[1]
    #
    #         i = concat[cur_batch, cur_elem]
    #         j = concat[cur_batch, cur_elem + vec_len]
    #         f = concat[cur_batch, cur_elem + 2*vec_len]
    #         o = concat[cur_batch, cur_elem + 3*vec_len]
    #         c_cur = c[cur_batch, cur_elem]
    #         new_c_cur = c_cur*sig(f + forget_bias) + sig(i) * tanh(j)
    #
    #         d_new_c_cur = d_new_c[cur_batch, cur_elem]
    #         d_new_h_cur = d_new_h[cur_batch, cur_elem]
    #
    #         d_concat = output_like(concat)
    #         d_c = output_like(c)
    #
    #         back_ch = d_new_c_cur + tanh_grad(new_c_cur)*sig(o)*d_new_h_cur
    #         d_i = tanh(j)*sig_grad(i)*back_ch
    #         d_j = sig(i)*tanh_grad(j)*back_ch
    #         d_f = c_cur*sig_grad(f+forget_bias)*back_ch
    #         d_c_cur = sig(f+forget_bias)*back_ch
    #         d_o = tanh(new_c_cur)*sig_grad(o)*d_new_h_cur
    #
    #         d_concat[cur_batch, cur_elem] = d_i
    #         d_concat[cur_batch, cur_elem+vec_len] = d_j
    #         d_concat[cur_batch, cur_elem+2*vec_len] = d_f
    #         d_concat[cur_batch, cur_elem+3*vec_len] = d_o
    #         d_c[pos] = d_c_cur
    #
    #         return d_concat, d_c
    #
    #
    # class TestLSTMGradient(unittest.TestCase):
    #     def test(self):
    #         print('*** Running Test: ' + self.__class__.__name__ + ' function: ' + _getframe().f_code.co_name)
    #         batches = 200
    #         vec_len = 500
    #         delta = 1e-3
    #         forget = 0.0
    #         adjoint_tests = 10
    #
    #         concat = np.random.normal(size=batches*4*vec_len).reshape((batches, 4*vec_len))
    #         c = np.random.normal(size=batches*vec_len).reshape((batches, vec_len))
    #
    #         delta_c = delta*np.random.normal(size=batches*vec_len).reshape((batches, vec_len))
    #         delta_concat = delta*np.random.normal(size=batches*4*vec_len).reshape((batches, 4*vec_len))
    #
    #         d_new_c = np.random.normal(size=batches*vec_len).reshape((batches, vec_len))
    #         d_new_h = np.random.normal(size=batches*vec_len).reshape((batches, vec_len))
    #
    #         # tf grad test
    #         with tf.Session() as sess:
    #             concat_tf = tf.Variable(concat)
    #             c_tf = tf.Variable(c)
    #             i, j, f, o = tf.split(1, 4, concat_tf)
    #
    #             new_c_tf = c_tf * tf.sigmoid(f + forget) + tf.sigmoid(i) * tf.tanh(j)
    #             new_h_tf = tf.tanh(new_c_tf) * tf.sigmoid(o)
    #
    #             new_c_ops, new_h_ops = LSTMP(concat_tf, c_tf, forget_bias=forget).as_tensorflow()
    #
    #             grad_tf = tf.gradients(new_c_tf, concat_tf)[0]
    #             grad_ops = tf.gradients(new_c_ops, concat_tf)[0]
    #
    #             sess.run(tf.initialize_all_variables())
    #             new_c_tf_eval, new_h_tf_eval = sess.run([new_c_tf, new_h_tf])
    #             new_c_ops_eval, new_h_ops_eval = sess.run([new_c_ops, new_h_ops])
    #
    #             grad_tf_eval = sess.run(grad_tf)
    #             grad_ops_eval = sess.run(grad_ops)
    #
    #         assert np.allclose(new_c_tf_eval, new_c_ops_eval)
    #         assert np.allclose(new_h_tf_eval, new_h_ops_eval)
    #
    #         assert np.allclose(grad_tf_eval, grad_ops_eval)
    #
    #         # fwd test
    #         with tf.Session() as sess:
    #             i, j, f, o = tf.split(1, 4, concat)
    #
    #             new_c_tf = c * tf.sigmoid(f + forget) + tf.sigmoid(i) * tf.tanh(j)
    #             new_h_tf = tf.tanh(new_c_tf) * tf.sigmoid(o)
    #
    #             new_c_eval, new_h_eval = sess.run([new_c_tf, new_h_tf])
    #
    #         new_c_ops, new_h_ops = LSTMP(concat, c, forget_bias=forget).evaluate_c()
    #         assert np.allclose(new_c_eval, new_c_ops)
    #         assert np.allclose(new_h_eval, new_h_ops)
    #
    #         # jacobian test
    #         new_c0, new_h0 = LSTMP(concat, c, forget_bias=forget).evaluate_c()
    #         new_c1, new_h1 = LSTMP(concat+delta_concat, c+delta_c, forget_bias=forget).evaluate_c()
    #         d_c, d_h = LSTMP_jacobian(concat, c, delta_concat, delta_c, forget_bias=forget).evaluate_c()
    #
    #         if not np.allclose(d_c, new_c1-new_c0, rtol=1e-2, atol=1e-6):
    #             comp = np.isclose(d_c, new_c1-new_c0, rtol=1e-2, atol=1e-6)
    #             assert np.sum(comp).astype(np.float32)/(batches*vec_len) > 0.999
    #         if not np.allclose(d_h, new_h1-new_h0, rtol=1e-2, atol=1e-6):
    #             comp = np.isclose(d_h, new_h1-new_h0, rtol=1e-2, atol=1e-6)
    #             assert np.sum(comp).astype(np.float32)/(batches*vec_len) > 0.999
    #
    #         # jacobian adjoint test
    #         jacobian = LSTMP_jacobian(concat, c, delta_concat, delta_c, forget_bias=forget)
    #         jacobian_adjoint = LSTMP_jacobian_adjoint(concat, c, d_new_c, d_new_h, forget_bias=forget)
    #
    #         lhs = []
    #         rhs = []
    #         for i in range(adjoint_tests):
    #             np.copyto(concat, np.random.normal(size=batches*4*vec_len).reshape((batches, 4*vec_len)))
    #             np.copyto(c, np.random.normal(size=batches*vec_len).reshape((batches, vec_len)))
    #
    #             np.copyto(delta_concat, delta*np.random.normal(size=batches*4*vec_len).reshape((batches, 4*vec_len)))
    #
    #             np.copyto(d_new_c, np.random.normal(size=batches*vec_len).reshape((batches, vec_len)))
    #             np.copyto(d_new_h, np.random.normal(size=batches*vec_len).reshape((batches, vec_len)))
    #
    #             d_c, d_h = jacobian.evaluate_c()
    #             d_concat_r, d_c_r = jacobian_adjoint.evaluate_c()
    #
    #             lhs.append(np.sum(d_new_h*d_h) + np.sum(d_new_c*d_c))
    #             rhs.append(np.sum(delta_c*d_c_r) + np.sum(delta_concat*d_concat_r))
    #
    #         assert np.allclose(np.array(lhs), np.array(rhs))
    #
#             np.copyto(delta_c, delta*np.random.normal(size=batches*vec_len).reshape((batches, vec_len)))
