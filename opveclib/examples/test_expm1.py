# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

"""
This example implements and tests the expm1 operator
- ie. element-wise exp(x) - 1,
which is equivalent to that defined for numpy.
"""

import unittest
import numpy as np
import opveclib as ovl
import tensorflow as tf

@ovl.operator()
def expm1(x):
    """
    Define the expm1 operator by defining the its operator function to be

    .. math::
      out_{i} = exp(x_{i}) - 1.0

    :param x: The input tensor
    :return: Element-wise exp(x) - 1

    :Examples:

    .. doctest::

        >>> import numpy as np
        >>> from opveclib import evaluate
        >>> from opveclib.examples import expm1
        >>> a = np.array([1e-10, -1e-10])
        >>> evaluate(expm1(a))
        array([  1.00000000e-10,  -1.00000000e-10])
        >>> np.expm1(a)
        array([  1.00000000e-10,  -1.00000000e-10])
        >>> ones = np.ones_like(a)
        >>> np.exp(a) - ones
        array([  1.00000008e-10,  -1.00000008e-10])
    """
    output = ovl.output_like(x)
    pos = ovl.position_in(x.shape)
    e = ovl.exp(x[pos])

    # note, this is an example of the use of the OVL conditional operators
    with ovl.if_(ovl.logical_and(ovl.isinf(x[pos]), x[pos] > 0.0)):
        output[pos] = x[pos]
    with ovl.elif_(e == 1.0):
        output[pos] = x[pos]
    with ovl.elif_ ((e - 1.0) == -1.0):
        output[pos] = -1.0
    with ovl.else_():
        output[pos] = (e - 1.0) * x[pos] / ovl.log(e)
    return output

@ovl.gradient(expm1)
@ovl.operator()
def expm1_grad(x, grad):
    """
    Define the expm1 gradient operator by defining the its operator function to be

    .. math::
      out_{i} = exp(x_{i}) * grad_{i}

    :param x: The input tensor argument
    :param grad: The input gradient tensor to the gradient operator
    :return: Element-wise gradient of the original operator
    """
    out = ovl.output(x.shape, x.dtype)
    pos = ovl.position_in(x.shape)
    out[pos] = ovl.exp(x[pos]) * grad[pos]
    return out

class TestExpm1(unittest.TestCase):
    ovl.clear_op_cache()
    def test(self):
        """
        Test the correctness of ovl operator vs numpy implementation
        """
        a = np.array([1e-10, -1e-10, 0.0, np.Infinity], dtype=np.float64)
        expm1_op = expm1(a)
        ref = np.expm1(a)
        ovl_res = ovl.evaluate(expm1_op)
        ovl.logger.info(u'numpy: ' + str(ref) + u' ovl: ' + str(ovl_res))
        assert np.allclose(ref, ovl_res, rtol=0, atol=1e-20)
        if ovl.cuda_enabled:
            assert np.allclose(np.expm1(a),
                      ovl.evaluate(expm1_op, target_language='cuda'),
                      rtol=0, atol=1e-20)

        # test  vs tensorflow
        # ensure TF runs on GPU when asked
        test_config=tf.ConfigProto(allow_soft_placement=False)
        test_config.graph_options.optimizer_options.opt_level = -1
        ones = np.ones_like(a)
        if ovl.cuda_enabled:
            devices = ['/cpu:0', '/gpu:0']
        else:
            devices = ['/cpu:0']
        with tf.Session(config=test_config) as sess:
           for dev_string in devices:
                with tf.device(dev_string):
                    expm1_tf = ovl.as_tensorflow(expm1_op)
                    sess.run(tf.initialize_all_variables())
                    expm1_tf_result = sess.run(expm1_tf)
                    assert np.allclose(ref, expm1_tf_result,
                                       rtol=0, atol=1e-20)

                    # TF exp - 1
                    tf_out = tf.exp(a) - ones
                    tf_result = tf_out.eval()
                    # this should fail
                    assert (np.allclose(ref, tf_result,
                                        rtol=0, atol=1e-20) == False)
        sess.close()

    def test_gradient(self):
        """
        Test the correctness of the gradient against tensorflow
        """
        if ovl.cuda_enabled:
            devices = ['/cpu:0', '/gpu:0']
        else:
            devices = ['/cpu:0']
        # ensure TF runs on GPU when asked
        test_config=tf.ConfigProto(allow_soft_placement=False)
        test_config.graph_options.optimizer_options.opt_level = -1
        with tf.Session(config=test_config) as sess:
           for dev_string in devices:
                with tf.device(dev_string):
                    a = np.random.random(100)
                    grad_input = tf.constant(np.random.random(100))
                    arg = tf.constant(a)
                    ovl_op = expm1(arg)
                    ones = tf.constant(np.ones_like(a))
                    ovl_out = ovl.as_tensorflow(ovl_op)
                    tf_out = tf.exp(arg) - ones

                    ovl_grad = tf.gradients(ovl_out, arg, grad_input)[0]
                    tf_grad = tf.gradients(tf_out, arg, grad_input)[0]
                    ovl_out, tf_out, ovl_grad, tf_grad = sess.run([ovl_out, tf_out, ovl_grad, tf_grad])

                    assert np.allclose(ovl_out, tf_out)
                    assert np.allclose(ovl_grad, tf_grad)
        sess.close()

