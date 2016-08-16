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
This example implements and tests the log1p operator
- ie. element-wise log(1 + x),
which is equivalent to that defined for numpy.
"""

import unittest
import numpy as np
import opveclib as ovl
import tensorflow as tf

@ovl.operator()
def log1p(x):
    """
    Define the log1p operator by defining the its operator function to be

    .. math::
      out_{i} = log(1.0 + x_{i})

    :param x: The input tensor
    :return: Element-wise log(1 + x)

    :Examples:

    .. doctest::

        >>> import numpy as np
        >>> from opveclib import evaluate
        >>> from opveclib.examples import log1p
        >>> a = np.array([1e-99, -1e-99])
        >>> evaluate(log1p(a))
        array([  1.00000000e-99,  -1.00000000e-99])
        >>> np.log1p(a)
        array([  1.00000000e-99,  -1.00000000e-99])
        >>> ones = np.ones_like(a)
        >>> np.log(ones + a)
        array([ 0.,  0.])
    """
    output = ovl.output_like(x)
    pos = ovl.position_in(x.shape)
    u = 1.0 + x[pos]
    d = u - 1.0

    # note, this is an example of the use of the OVL conditional operators
    with ovl.if_(d == 0):
        output[pos] = x[pos]
    with ovl.else_():
        output[pos] = ovl.log(u) * x[pos] / d
    return output

class TestExpm1(unittest.TestCase):
    ovl.clear_op_cache()
    def test(self):
        """
        Test the correctness of ovl operator vs numpy implementation
        """
        a = np.array([1e-99, -1e-99, 0.0], dtype=np.float64)
        log1pOp = log1p(a)
        ref = np.log1p(a)
        ovl_res = ovl.evaluate(log1pOp)
        ovl.logger.debug(u'numpy: ' + str(ref) + u' ovl: ' + str(ovl_res))
        assert np.allclose(ref, ovl_res, rtol=0, atol=1e-20)
        if ovl.cuda_enabled:
            assert np.allclose(np.log1p(a),
                      ovl.evaluate(log1pOp, target_language='cuda'),
                      rtol=0, atol=1e-20)

        # test  vs tensorflow
        test_config=tf.ConfigProto(allow_soft_placement=False)
        # ensure TF runs on GPU when asked
        test_config.graph_options.optimizer_options.opt_level = -1
        ones = np.ones_like(a)
        if ovl.cuda_enabled:
            devices = ['/cpu:0', '/gpu:0']
        else:
            devices = ['/cpu:0']
        with tf.Session(config=test_config) as sess:
           for dev_string in devices:
                with tf.device(dev_string):
                    log1p_tf = ovl.as_tensorflow(log1pOp)
                    sess.run(tf.initialize_all_variables())
                    log1p_tf_result = sess.run(log1p_tf)
                    assert np.allclose(ref, log1p_tf_result,
                                       rtol=0, atol=1e-20)

                    # TF exp - 1
                    tf_out = tf.log(a - ones)
                    tf_result = tf_out.eval()
                    # this should fail
                    assert (np.allclose(ref, tf_result,
                                        rtol=0, atol=1e-20) == False)
        sess.close()
