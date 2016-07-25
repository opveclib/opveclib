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
from ..operator import operator, gradient, as_tensorflow
from ..expression import position_in, output_like, if_, elif_, else_, exp, logical_and, variable, tanh
from ..local import clear_op_cache


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


@operator()
def clip_grad_op(arg, d_arg, threshold1=None, threshold2=None):
    assert arg.tensor_type == d_arg.tensor_type

    pos = position_in(arg.shape)
    g = output_like(arg)

    arg_at_pos = arg[pos]

    with if_(arg_at_pos < threshold1):
        g[pos] = 0
    with elif_(arg_at_pos > threshold2):
        g[pos] = 0
    with else_():
        g[pos] = d_arg[pos]

    return g


@gradient(clip)
def clip_grad(arg, d_arg, threshold1=None, threshold2=None):
    return clip_grad_op(arg, d_arg, threshold1=threshold1, threshold2=threshold2)


@operator()
def sigmoid(arg):
    out = output_like(arg)
    pos = position_in(arg.shape)
    out[pos] = 1/(1 + exp(-arg[pos]))

    return out


@operator()
def sigmoid_grad_op(arg, grad_above):
    out = output_like(arg)
    pos = position_in(arg.shape)

    valid_grad = logical_and(arg[pos] > -50, arg[pos] < 50)
    result = variable(0, arg.dtype)
    with if_(valid_grad):
        e = exp(-arg[pos])
        result <<= e/((1+e)*(1+e))

    out[pos] = result*grad_above[pos]

    return out


@gradient(sigmoid)
def sigmoid_grad(arg, d_out):
    return sigmoid_grad_op(arg, d_out)


class TestGradients(unittest.TestCase):
    # clear_op_cache()

    def test_clip_grad(self):
        """
        Define a simple clip function and its gradient and test against a tensorflow reference
        """

        with tf.Session() as s:
            a = tf.constant(np.random.random(10))
            clip_ovl = as_tensorflow(clip(a, threshold1=0.1, threshold2=0.9))
            sum_ovl = tf.reduce_sum(clip_ovl)
            clip_tf = tf.clip_by_value(a, 0.1, 0.9)
            sum_tf = tf.reduce_sum(clip_tf)
            grad_above = np.random.random(1)

            clip_grad_ovl = tf.gradients(sum_ovl, a, grad_ys=grad_above)[0]
            clip_grad_tf = tf.gradients(sum_tf, a, grad_ys=grad_above)[0]

            clip_grad_ovl, clip_grad_tf = s.run([clip_grad_ovl, clip_grad_tf])
            assert np.all(np.equal(clip_grad_ovl, clip_grad_tf))

    def test_sigmoid_grad(self):
        with tf.Session() as s:
            num_points = 10
            a = tf.constant(np.random.random(num_points))
            sig_ovl = as_tensorflow(sigmoid(a))
            sig_tf = tf.sigmoid(a)
            grad_above = np.random.random(num_points)

            grad_ovl = tf.gradients(sig_ovl, a, grad_ys=grad_above)[0]
            grad_tf = tf.gradients(sig_tf, a, grad_ys=grad_above)[0]

            sig_ovl, sig_tf, grad_ovl, grad_tf = s.run([sig_ovl, sig_tf, grad_ovl, grad_tf])
            assert np.all(np.equal(sig_ovl, sig_tf))
            assert np.allclose(grad_ovl, grad_tf)


        # @operator()
        # def tanh_grad(arg):
        #     t = tanh(arg)
        #     return 1 - t*t

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
