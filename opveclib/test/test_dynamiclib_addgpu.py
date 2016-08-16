# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import tensorflow as tf
import numpy as np
import os
import unittest
import subprocess
from ..operator import _DynamicLibOp, _default_cuda_threads_per_block
from ..local import cuda_enabled, cuda_directory, cache_directory, clear_op_cache, logger, cxx


# Test to ensure valid calculation on both CPU and GPU.
# All GPU calculations have an extra 1.0 added to each element to verify that
# code is actually running on GPU
class DynamicLibAddGPUTest(unittest.TestCase):
    clear_op_cache()

    def test1(self):

        # build the operator libs if needed
        cpulib = os.path.join(cache_directory, "libaddcpu.so")
        gpulib = os.path.join(cache_directory, "libaddgpu.so")
        if not os.path.exists(cpulib):
            this_file_path = os.path.abspath(__file__)
            this_directory = os.path.split(this_file_path)[0]

            cpp_path = os.path.join(this_directory, 'addcpu.cpp')
            subprocess.call([cxx, '-fPIC', '-Wall',
                         '-std=c++11', '-Ofast', '-Wextra',
                         '-g', '-pedantic',
                         '-I'+this_directory+'/..',
                         '-o', cpulib, '-shared',  cpp_path])

        if cuda_enabled:
            if not os.path.exists(gpulib):
                this_file_path = os.path.abspath(__file__)
                this_directory = os.path.split(this_file_path)[0]

                nvcc_path = os.path.join(cuda_directory, 'bin/nvcc')
                cuda_path = os.path.join(this_directory, 'addgpu.cu')
                cuda_o_path = os.path.join(cache_directory, 'addgpu.o')

                subprocess.call([nvcc_path, '-O3', '--use_fast_math', '--relocatable-device-code=true', '--compile', '-Xcompiler',
                                '-fPIC', '-std=c++11', '-I'+this_directory+'/..',
                                 cuda_path, '-o', cuda_o_path])
                subprocess.call([nvcc_path, '-shared', '-o', gpulib, cuda_o_path])
                # clean up .o files
                subprocess.call(['rm', cuda_o_path])

            devices = ['/cpu:0', '/gpu:0']
        else:
            devices = ['/cpu:0']
        for dev_string in devices:
            logger.debug('*** device: {dev}'.format(dev= dev_string))
            test_config=tf.ConfigProto(allow_soft_placement=False)
            # Don't perform optimizations for tests so we don't inadvertently run
            # gpu ops on cpu
            test_config.graph_options.optimizer_options.opt_level = -1
            with tf.Session(config=test_config):
                logger.debug('*** add2float')
                with tf.device(dev_string):
                    in0 = np.random.rand(3,50).astype(np.float32)
                    in1 = np.random.rand(3,50).astype(np.float32)
                    ones = np.ones((3,50), dtype=np.float32)
                    output = _DynamicLibOp.module().dynamic_lib(inputs=[in0, in1],
                                                               out_shapes=[[3,50]],
                                                               out_types=['float'],
                                                               cpu_lib_path=cpulib,
                                                               cpu_func_name="add2float",
                                                               gpu_lib_path=gpulib,
                                                               gpu_func_name="add2float",
                                                               serialized_grad_dag='',
                                                               grad_dag_arg_index=[],
                                                               cuda_threads_per_block=_default_cuda_threads_per_block)

                    ref = np.add(in0,in1)
                    if (dev_string is '/gpu:0'):
                        ref = np.add(ref,ones)
                    assert np.allclose(output[0].eval(), ref)

                    in2 = np.random.rand(3,50).astype(np.float64)
                    logger.debug('*** addFloatDoubleFloat')
                    output = _DynamicLibOp.module().dynamic_lib(inputs=[in0, in2, in1],
                                                                       out_shapes=[[3,50]],
                                                                       out_types=['float'],
                                                                       cpu_lib_path= cpulib,
                                                                       cpu_func_name="addFloatDoubleFloat",
                                                                       gpu_lib_path= gpulib,
                                                                       gpu_func_name="addFloatDoubleFloat",
                                                                       serialized_grad_dag='',
                                                                       grad_dag_arg_index=[],
                                                                       cuda_threads_per_block=_default_cuda_threads_per_block)
                    ref = (in0 + in2 + in1).astype(np.float32)
                    if (dev_string is '/gpu:0'):
                        ref = ref + ones
                    assert np.allclose(output[0].eval(), ref)

                    logger.debug('*** sumAndSq')
                    output = _DynamicLibOp.module().dynamic_lib(inputs=[in0, in2],
                                                                       out_shapes=[[3,50], [3,50]],
                                                                       out_types=['float', 'float'],
                                                                       cpu_lib_path= cpulib,
                                                                       cpu_func_name="sumAndSq",
                                                                       gpu_lib_path= gpulib,
                                                                       gpu_func_name="sumAndSq",
                                                                       serialized_grad_dag='',
                                                                       grad_dag_arg_index=[],
                                                                       cuda_threads_per_block=_default_cuda_threads_per_block)

                    out0 = (in0 + in2).astype(np.float32)
                    if (dev_string is '/gpu:0'):
                        out0 = out0 + ones
                    out1 = np.multiply(out0, out0)
                    if (dev_string is '/gpu:0'):
                        out1 = out1 + ones
                    assert np.allclose(output[0].eval(), out0)
                    assert np.allclose(output[1].eval(), out1)

                    # make sure we can also use a standard TF gpu operator in the same session
                    logger.debug('*** TF numerics op')
                    x_shape = [5, 4]
                    x = np.random.random_sample(x_shape).astype(np.float32)
                    t = tf.constant(x, shape=x_shape, dtype=tf.float32)
                    t_verified = tf.verify_tensor_all_finite(t, "Input is not a number.")
                    assert np.allclose(x, t_verified.eval())
