
# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import ctypes
import errno
import hashlib
import os
import inspect
import subprocess

import tensorflow as tf
import numpy as np
from numpy.ctypeslib import ndpointer

from .expression import TensorType, ExpressionDAG, input, float32, float64, OutputTensor
from .local import version, cache_directory, cuda_enabled, cuda_directory


class _TensorParam(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p),
                ("dtype", ctypes.c_int),
                ("len", ctypes.c_size_t)]


class Operator(object):
    """
    Class which is extended to define a new operator and its gradient.
    """
    _inference_registered = False
    _dynamiclibop_module = None
    _gradient_registered = False
    _conversion_registered = False
    _default_cuda_threads_per_block = 64

    @staticmethod
    def _register_shape_inference():
        if Operator._inference_registered is False:
            Operator._inference_registered = True

            @tf.RegisterShape("DynamicLib")
            def _tensor_ops_shape(op):
                return op.get_attr('out_shapes')

    @staticmethod
    def _load_dynamiclib_module():
        if Operator._dynamiclibop_module is None:
            libname = 'dynamiclibop.so.' + version
            dynamiclibop_path = os.path.join(cache_directory, libname)
            if not os.path.exists(dynamiclibop_path):
                # build the library if it does not exist already
                tf_include = tf.sysconfig.get_include()
                # resolve the directory of this file
                this_file_path = os.path.abspath(__file__)
                this_directory = os.path.split(this_file_path)[0]
                try:
                    if cuda_enabled:
                        tf.logging.log(tf.logging.INFO, '*** building dynamiclibop for GPU')
                        subprocess.check_output(['g++', '-fPIC', '-Wall', '-shared',
                                         '-std=c++11', '-O2', '-Wextra', '-DGOOGLE_CUDA=1',
                                         '-o', dynamiclibop_path,
                                         this_directory + '/dynamiclibop.cc',
                                         '-isystem', cuda_directory + '/include',
                                         '-isystem', tf_include],
                                          stderr=subprocess.STDOUT,
                                          universal_newlines=True)
                    else:
                        tf.logging.log(tf.logging.INFO, '*** building dynamiclibop for CPU')
                        subprocess.check_output(['g++', '-fPIC', '-Wall', '-shared',
                                         '-std=c++11', '-O2', '-Wextra',
                                         '-o', dynamiclibop_path,
                                         this_directory + '/dynamiclibop.cc',
                                         '-isystem', tf_include],
                                          stderr=subprocess.STDOUT,
                                          universal_newlines=True)
                except subprocess.CalledProcessError as exception:
                    tf.logging.log(tf.logging.ERROR, 'g++ error: ' + exception.output)
                    raise

            Operator._dynamiclibop_module = tf.load_op_library(dynamiclibop_path)

    @staticmethod
    def _register_gradient():
        if not Operator._gradient_registered:
            Operator._gradient_registered = True

            from tensorflow.python.framework import ops as tf_ops

            @tf_ops.RegisterGradient("DynamicLib")
            def _dynamic_lib_grad(op, *grads_above):
                num_inputs = len(op.inputs)

                gpu_grad_name = op.get_attr('gpu_grad_func_name')
                gpu_grad_lib = op.get_attr('gpu_grad_lib_path')
                cpu_grad_name = op.get_attr('cpu_grad_func_name')
                cpu_grad_lib = op.get_attr('cpu_grad_lib_path')
                cuda_threads_per_block = op.get_attr('cuda_threads_per_block')

                if cpu_grad_name == '':
                    grads = []
                    for i in range(num_inputs):
                        grads.append(None)

                    return grads
                else:
                    out_shapes = []
                    out_types = []
                    for cur_input in op.inputs:
                        cur_type = TensorType.like(cur_input)
                        if cur_type.dtype == float32:
                            tf_type = 'float'
                        elif cur_type.dtype == float64:
                            tf_type = 'double'
                        else:
                            raise NotImplementedError('Only floats and doubles currently supported.')

                        out_types.append(tf_type)
                        out_shapes.append(cur_type.shape)

                    inputs = []
                    for inp in op.inputs:
                        inputs.append(inp)

                    try:
                        len(grads_above)
                    except TypeError:
                        inputs.append(grads_above)
                    else:
                        for grad_above in list(grads_above):
                            inputs.append(grad_above)

                    grads = Operator._dynamiclibop_module.dynamic_lib(inputs=inputs,
                                                                      out_shapes=out_shapes,
                                                                      out_types=out_types,
                                                                      cpu_lib_path=cpu_grad_lib,
                                                                      cpu_func_name=cpu_grad_name,
                                                                      gpu_lib_path=gpu_grad_lib,
                                                                      gpu_func_name=gpu_grad_name,
                                                                      gpu_grad_func_name='',
                                                                      gpu_grad_lib_path='',
                                                                      cpu_grad_func_name='',
                                                                      cpu_grad_lib_path='',
                                                                      cuda_threads_per_block=cuda_threads_per_block)
                    return grads

    # @staticmethod
    # def _register_conversion():
    #
    #     if not Operation._conversion_registered:
    #         import tensorflow as tf
    #
    #         def conv(value, dtype=None, name=None, as_ref=False):
    #             if len(value.output_types) != 1:
    #                 raise ValueError('Cannot implicitly convert multi-output Ops.')
    #
    #             output = value.as_tensorflow()
    #
    #             if as_ref:
    #                 raise NotImplementedError()
    #
    #             if dtype is not None:
    #                 raise NotImplementedError()
    #
    #             return output
    #
    #         tf.register_tensor_conversion_function(Operation, conv)
    #         Operation._conversion_registered = True

    @staticmethod
    def _make_generic_c(src, name):
        # look for generic c++ shared library in the operator cache
        generic_cpp_so_path = os.path.join(cache_directory, name + '_generic_cpp.so')
        if not os.path.exists(generic_cpp_so_path):
            generic_cpp_path = os.path.join(cache_directory, name + '_generic_cpp.cpp')
            with open(generic_cpp_path, 'w') as f:
                f.write(src)

            this_file_path = os.path.abspath(__file__)
            this_directory = os.path.split(this_file_path)[0]
            try:
                subprocess.check_output(['g++', '-fPIC', '-std=c++11', '-g', '-pedantic',
                             '-Wall', '-Wextra',
                             '-I'+this_directory,
                             '-shared',
                             '-o', generic_cpp_so_path, generic_cpp_path],
                              stderr=subprocess.STDOUT,
                              universal_newlines=True)
            except subprocess.CalledProcessError as exception:
                tf.logging.log(tf.logging.ERROR, 'g++ error: ' + exception.output)
                raise

        return generic_cpp_so_path

    @staticmethod
    def _make_generic_cuda(src, name):
        # look for generic cuda shared library in the operator cache
        generic_cuda_so_path = os.path.join(cache_directory, name + '_generic_cuda.so')
        if not os.path.exists(generic_cuda_so_path):
            # generate and compile generic cuda operator
            nvcc_path = os.path.join(cuda_directory, 'bin/nvcc')
            generic_cuda_path = os.path.join(cache_directory, name + '_generic_cuda.cu')
            generic_cuda_o_path = os.path.join(cache_directory, name + '_generic_cuda.o')

            with open(generic_cuda_path, 'w') as f:
                f.write(src)

            this_file_path = os.path.abspath(__file__)
            this_directory = os.path.split(this_file_path)[0]
            try:
                subprocess.check_output([nvcc_path, '-O3', '--use_fast_math', '--relocatable-device-code=true', '--compile',
                             '-Xcompiler', '-fPIC', '-std=c++11', '-I'+this_directory,
                             generic_cuda_path, '-o', generic_cuda_o_path],
                             stderr=subprocess.STDOUT,
                             universal_newlines=True)
                subprocess.check_output([nvcc_path, '-shared', '-o', generic_cuda_so_path, generic_cuda_o_path],
                             stderr=subprocess.STDOUT,
                             universal_newlines=True)
            except subprocess.CalledProcessError as exception:
                tf.logging.log(tf.logging.ERROR, 'nvcc error: ' + exception.output)
                raise

            # clean up .o files
            subprocess.call(['rm', generic_cuda_o_path])

        return generic_cuda_so_path

    @staticmethod
    def _unwrap_single(x):
        if len(x) is 1:
            return x[0]
        else:
            return x

    def __init__(self, *inputs, **kwargs):
        # set default options
        self._options = kwargs

        def set_default_option(options, name, value):
            if name not in options:
                options[name] = value

        tf.logging.log(tf.logging.DEBUG, 'Creating Op ' + self.__class__.__name__)

        set_default_option(self._options, 'verbose', False)
        set_default_option(self._options, 'clear_cache', False)

        self._inputs = list(inputs)

        self._input_types = []
        for inp_n, inp in enumerate(inputs):
            try:
                self._input_types.append(TensorType.like(inp))
            except AttributeError:
                raise TypeError('Received a ' + inp.__class__.__name__ + ' instead of a tensor at argument position ' +
                                str(inp_n + 1) + ' in the Op constructor. ' +
                                'Should this argument be passed as a constant (keyword argument) instead?')

        def interpret_function(input_types, function):
            f_name = function.__name__
            num_inputs = len(input_types)

            # parse arg spec of the function and build up a dictionary of defaults to be applied if constants
            # of the same name are not passed to contructor
            arg_spec = inspect.getargspec(function).args[1:]
            defaults = inspect.getargspec(function).defaults
            defaults_dict = {}
            if defaults is not None:
                for default_n, cur_default in enumerate(defaults):
                    defaults_dict[arg_spec[default_n-len(defaults)]] = cur_default

            # keep track of which names are present in the constants dict or the defaults dict.
            input_names = []
            constant_names = []
            default_names = []
            for arg in arg_spec:
                if arg in list(self._options.keys()):
                    constant_names.append(arg)
                elif arg in list(defaults_dict.keys()):
                    default_names.append(arg)
                else:
                    input_names.append(arg)

            # raise an error if the number of non-keyword args in the constructor is different from the number of
            # non-constant inputs to the function
            if len(input_names) != num_inputs:
                err_msg = '\n'
                additional = ''
                if len(constant_names) > 0:
                    err_msg += f_name + ' function received ' + str(len(constant_names)) + ' constants:\n' + \
                               str(constant_names) + '\n'
                    additional = ' additional'

                err_msg += 'Based on constructor call pattern, ' + f_name + ' function signature expects ' + \
                           str(len(input_names)) + additional + \
                           ' input tensor argument(s):\n' + str(input_names) + '\n'
                err_msg += 'but was supplied with ' + str(num_inputs) + '.\n'
                if len(input_names) > num_inputs:
                    remaining_names = input_names[num_inputs - len(input_names):]
                    err_msg += 'Should ' + str(remaining_names) + ' be passed to constructor as constant?'
                raise TypeError(err_msg)

            ExpressionDAG.clear()

            # create input expressions
            input_exprs = []
            for cur_type in input_types:
                input_exprs.append(input(cur_type))

            args = []
            expr_n = 0
            for cur_arg in arg_spec:
                if cur_arg in list(self._options.keys()):
                    args.append(self._options[cur_arg])
                elif cur_arg in defaults_dict:
                    args.append(defaults_dict[cur_arg])
                else:
                    args.append(input_exprs[expr_n])
                    expr_n += 1

            # interpret function to build up ExpressionDAG
            output_exprs = function(*args)
            if output_exprs is None:
                raise ValueError('No outputs returned from ' + f_name + ' function')

            # wrap as list if only one output
            try:
                len(output_exprs)
            except TypeError:
                output_exprs = [output_exprs]

            # make sure number of returned parameters equals the number of declared outputs
            if len(output_exprs) != ExpressionDAG.num_outputs:
                raise ValueError('Defined ' + str(ExpressionDAG.num_outputs) + ' outputs, but returned ' +
                                 str(len(output_exprs)) +
                                 '. Number of defined outputs must equal number of returned outputs')

            # make sure all returned values are output expressions
            # reorder output io_index according to return order instead of declaration order
            output_types = []
            prev_index = []
            for index, expr in enumerate(output_exprs):
                if type(expr) is not OutputTensor:
                    raise TypeError('User functions must only return outputs. Instead got:\n' + str(expr))
                prev_index.append(ExpressionDAG.expr_index(expr))
                expr.proto_expr.io_index = index
                output_types.append(TensorType.like(expr))
            # reorder declaration of outputs in expression dag
            prev_index.sort()
            for index, expr in zip(prev_index, output_exprs):
                ExpressionDAG.exprs[index] = expr
                ExpressionDAG.expr_ids[index] = id(expr)

            expression_dag = ExpressionDAG.as_proto()
            ExpressionDAG.clear()

            return output_types, expression_dag

        self.output_types, self.op_expression_dag = interpret_function(self._input_types, self.op)

        # define a function name based on the operator hash
        self.op_name = 'f' + hashlib.sha224(self.op_expression_dag.SerializeToString() + version.encode('utf-8')).hexdigest()

        tf.logging.log(tf.logging.DEBUG, 'Generating code for Op ' +
                       self.__class__.__name__ + ' as ' + self.op_name)

        # generate code of the operator
        self.op_c_src, self.op_cuda_src, self.op_cuda_launch_template, self.op_c_generic, self.op_cuda_generic = \
            ExpressionDAG.generate(self.op_expression_dag, self.op_name)

        # define the c types for op input and output arguments
        self.op_argtypes = []
        for in_cur in self._input_types:
            t = in_cur.dtype.as_ctypes()
            p = ndpointer(t, flags="C_CONTIGUOUS")
            self.op_argtypes.append(p)

        for out_cur in self.output_types:
            t = out_cur.dtype.as_ctypes()
            p = ndpointer(t, flags="C_CONTIGUOUS")
            self.op_argtypes.append(p)

        # parse grad function
        try:
            self.grad()

        # grad not defined
        except ValueError:
            self.grad_expression_dag = None
            self.grad_name = None
            self.grad_c_src = None
            self.grad_cuda_src = None
            self.grad_cuda_launch_template = None
            self.grad_c_generic = None
            self.grad_cuda_generic = None
            self.grad_argtypes = None
        except TypeError:
            tf.logging.log(tf.logging.DEBUG, 'Creating gradient for Op ' + self.__class__.__name__)
            grad_arg_spec = inspect.getargspec(self.grad).args[1:]

            # make sure initial part of gradient function signature matches op function signature
            for arg_n, op_arg in enumerate(inspect.getargspec(self.op).args[1:]):
                if op_arg != grad_arg_spec[arg_n]:
                    raise TypeError('Gradient function must have same initial argument names as the op function. ' +
                                    'Expected arg "' + str(op_arg) + '", but got "' + str(grad_arg_spec[arg_n]) + '".')
            grad_input_types = []
            for t in self._input_types:
                grad_input_types.append(TensorType.like(t))
            for t in self.output_types:
                grad_input_types.append(TensorType.like(t))

            grad_types, self.grad_expression_dag = interpret_function(grad_input_types, self.grad)

            for grad_n, grad_type in enumerate(grad_types):
                if grad_type != self._input_types[grad_n]:
                    raise TypeError('Gradient function must output tensor list with a types identical '
                                    'to the op functions inputs.')

            self.grad_name = 'f' + hashlib.sha224(self.grad_expression_dag.SerializeToString() + version.encode('utf-8')).hexdigest()
            tf.logging.log(tf.logging.DEBUG, 'Generating code for gradient for Op ' +
                           self.__class__.__name__ + ' as ' + self.grad_name)
            self.grad_c_src, self.grad_cuda_src, self.grad_cuda_launch_template, self.grad_c_generic, \
                self.grad_cuda_generic = ExpressionDAG.generate(self.grad_expression_dag, self.grad_name)

            # define c types of grad arguments
            self.grad_argtypes = []
            for at in self.op_argtypes:
                self.grad_argtypes.append(at)

            for in_cur in self._input_types:
                t = in_cur.dtype.as_ctypes()
                p = ndpointer(t, flags="C_CONTIGUOUS")
                self.grad_argtypes.append(p)

        else:
            raise TypeError('Badly formed gradient function. Gradient function requires arguments.')

        # create cache directory if it doesn't already exist
        try:
            os.makedirs(cache_directory)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        # clear cache of all files related to this operator
        if self._options['clear_cache']:
            for filename in os.listdir(cache_directory):
                if self.op_name in filename:
                    os.remove(os.path.join(cache_directory, filename))
                if self.grad_name is not None and self.grad_name in filename:
                    os.remove(os.path.join(cache_directory, filename))

        tf.logging.log(tf.logging.DEBUG, 'Finished creating Op ' + self.__class__.__name__)

        # initialize lazily defined functions and buffers used by evaluation infrastructure
        self._op_c_function = None
        self._op_cuda_function = None
        self._output_buffers = None
        self._output_params = None
        self._input_params = None
        self._active_eval_fcn = None

        self._test_cuda_op = None
        self._test_c_op = None

    def op(self, *input_tensors, **constants):
        """
        Abstract member that must be implemented to define an operator

        :param input_tensors: tensor arguments
        :param constants: constant arguments, passed in by keyword
        :return: Must return a list of output tensors, defined by this function
        """
        raise NotImplementedError("Abstract class")

    def grad(self, *inputs):
        raise ValueError()

    def _define_eval_params(self, lib, fcn_name):

        # determine input parameters
        if self._input_params is None:

            for inp in self._inputs:
                if not isinstance(inp, np.ndarray):
                    raise SyntaxError('Can only evaluate operators when the inputs are numpy arrays.')

            inputs = []
            for in_t, in_data in zip(self._input_types, self._inputs):
                inputs.append(_TensorParam(data=in_data.ctypes.data,
                                           dtype=ctypes.c_int(in_t.dtype.proto_dtype),
                                           len=ctypes.c_size_t(in_t.size)))
            self._input_params = np.array(inputs, dtype=_TensorParam)

        # allocate new buffers for output arrays and define output parameters
        if self._output_buffers is None:
            self._output_buffers = []
            for out_cur in self.output_types:
                t = out_cur.dtype.as_numpy()
                new_buff = np.empty(out_cur.shape, dtype=t)
                self._output_buffers.append(new_buff)

            outputs = []
            for out_t, out_array in zip(self.output_types, self._output_buffers):
                outputs.append(_TensorParam(data=out_array.ctypes.data,
                                            dtype=ctypes.c_int(out_t.dtype.proto_dtype),
                                            len=ctypes.c_size_t(out_t.size)))
            self._output_params = np.array(outputs, dtype=_TensorParam)

        # if changing lib functions, initialize with zeros to avoid carry-over from previous results
        if self._active_eval_fcn != lib+fcn_name:
            self._active_eval_fcn = lib+fcn_name
            for out in self._output_buffers:
                out[:] = 0

    @staticmethod
    def _check_proto():
        # build the protobuf header file. This must match the version of protoc
        # and libprotobuf-dev that is installed on the user system. Otherwise the generated file
        # will be incompatible with the protoc system header files.
        proto_header = os.path.join(cache_directory, 'language.pb.h')
        if not os.path.exists(proto_header):
            this_file_path = os.path.abspath(__file__)
            this_directory = os.path.split(this_file_path)[0]
            proto_path = os.path.join(this_directory, 'language.proto')

            try:
                subprocess.check_output(['protoc', proto_path, '--proto_path='+this_directory,
                             '--cpp_out='+cache_directory],
                             stderr=subprocess.STDOUT,
                             universal_newlines=True)
            except subprocess.CalledProcessError as exception:
                tf.logging.log(tf.logging.ERROR, 'protoc error: ' + exception.output)
                raise

    def evaluate_c(self, profiling_iterations=None):
        """
        Evaluate dthe compiled C code for this operator, mainly used for testing. This function uses a test operator
        function for running the generated generic version of the operator so it does not depend on an external
        execution runtime. This also means that this function only works for operators whose inputs are numpy arrays.

        :param profiling_iterations: Number of times to run this operator for profiling purposes.
            Must be a positive int.

        :return:  If profiling_iterations is set to None, returns the numpy array, or list of numpy arrays if there are
            multiple outputs, containing results from evaluation. If profiling_iterations is set, returns a tuple of the
            output array(s), and a numpy array that contains the time, in ms, that each function evaluation took.
        """

        # get the C test function from it's .so (compiles if necessary)
        lib_path = Operator._make_generic_c(self.op_c_generic, self.op_name).encode('utf-8')
        fcn_name = (self.op_name + '_generic_cpp').encode('utf-8')

        self._define_eval_params(lib_path, fcn_name)

        if profiling_iterations is None:
            iters = 1
        else:
            if not isinstance(profiling_iterations, int) or profiling_iterations < 1:
                raise ValueError('Profiling iterations must be a positive int, but received: ' +
                                 str(profiling_iterations))
            iters = profiling_iterations

        eval_times_ms = np.empty(iters, dtype=np.float64)
        eval_times_ms[:] = np.nan

        num_inputs = len(self._input_types)
        num_outputs = len(self.output_types)

        if self._test_c_op is None:
            testlib_path = os.path.join(cache_directory, 'libtestcop.so.'+version)
            try:
                libtest = ctypes.cdll.LoadLibrary(testlib_path)
            except OSError:
                Operator._check_proto()
                this_file_path = os.path.abspath(__file__)
                this_directory = os.path.split(this_file_path)[0]

                # build the test framework library
                cc_path = os.path.join(this_directory, 'testcop.cc')

                try:
                    subprocess.check_output(['g++', '-fPIC', '-Wall', '-shared',
                                 '-std=c++11', '-Ofast', '-Wextra',
                                 '-I'+this_directory,
                                 '-I'+cache_directory,
                                 '-o', testlib_path, cc_path],
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True)
                except subprocess.CalledProcessError as exception:
                    tf.logging.log(tf.logging.ERROR, 'g++ error: ' + exception.output)
                    raise

                libtest = ctypes.cdll.LoadLibrary(testlib_path)

            self._test_c_op = libtest.testCOperator
            self._test_c_op.restype = ctypes.c_int16
            self._test_c_op.argtypes = \
                [ctypes.c_char_p, ctypes.c_char_p,
                 ndpointer(dtype=_TensorParam, flags="C_CONTIGUOUS"), ctypes.c_size_t,
                 ndpointer(dtype=_TensorParam, flags="C_CONTIGUOUS"), ctypes.c_size_t,
                 ndpointer(dtype=ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t]

        # run the operator
        err = self._test_c_op(lib_path, fcn_name,
                              self._input_params, ctypes.c_size_t(num_inputs),
                              self._output_params, ctypes.c_size_t(num_outputs),
                              eval_times_ms,
                              ctypes.c_size_t(iters))

        if err != 0 or np.isnan(eval_times_ms).any():
            tf.logging.log(tf.logging.ERROR, 'Test C operator failed for Op ' + self.__class__.__name__)
            raise ValueError('Test C operator failed for Op ' + self.__class__.__name__)

        if profiling_iterations is None:
            return Operator._unwrap_single(self._output_buffers)
        else:
            return Operator._unwrap_single(self._output_buffers), eval_times_ms

    def evaluate_cuda(self, cuda_threads_per_block=_default_cuda_threads_per_block, profiling_iterations=None):
        """
        Evaluate the compiled CUDA code for this operator, mainly used for testing. This function uses a test operator
        function for running the generated generic version of the operator so it does not depend on an external
        execution runtime. This also means that this function only works for operators whose inputs are numpy arrays.

        :param profiling_iterations: Number of times to run this operator for profiling purposes.
            Must be a positive int.
        :param cuda_threads_per_block: number of cuda threads to use

        :return: If profiling_iterations is set to None, returns the numpy array, or list of numpy arrays if there are
            multiple outputs, that results from evaluation. If profiling_iterations is set, returns a tuple of the
            output array(s), and a numpy array that contains the time, in ms, that each function evaluation took.
        """
        if not cuda_enabled:
            raise RuntimeError('CUDA is not enabled')

        # get the CUDA test function from it's .so (compiles if necessary)
        lib_path = Operator._make_generic_cuda(self.op_cuda_generic, self.op_name).encode('utf-8')
        fcn_name = (self.op_name + '_generic_cuda').encode('utf-8')
        self._define_eval_params(lib_path, fcn_name)

        num_inputs = len(self._input_types)
        num_outputs = len(self.output_types)

        if profiling_iterations is None:
            iters = 1
        else:
            if not isinstance(profiling_iterations, int) or profiling_iterations < 1:
                raise ValueError('Profiling iterations must be a positive int, but received: ' +
                                 str(profiling_iterations))
            iters = profiling_iterations

        eval_times_ms = np.empty(iters, dtype=np.float64)
        eval_times_ms[:] = np.nan

        # lazily compile testcudaop.cc
        if self._test_cuda_op is None:
            testlib_path = os.path.join(cache_directory, 'libtestcudaop.so.'+version)
            try:
                libtest = ctypes.cdll.LoadLibrary(testlib_path)
            except OSError:
                Operator._check_proto()
                this_file_path = os.path.abspath(__file__)
                this_directory = os.path.split(this_file_path)[0]

                # build the test framework library
                cc_path = os.path.join(this_directory, 'testcudaop.cc')
                o_path = os.path.join(cache_directory, 'testcudaop.o')
                nvcc_path = os.path.join(cuda_directory, 'bin/nvcc')
                try:
                    subprocess.check_output([nvcc_path, '-O3', '--relocatable-device-code=true',
                                 '-x', 'cu', '--compile', '-Xcompiler',
                                 '-fPIC', '-std=c++11',
                                 '-I'+this_directory,
                                 '-I'+cache_directory,
                                 cc_path, '-o', o_path],
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True)

                    # relocatable device code has to be defined when linking in addition
                    # to compiling. g++ has no concept of this, so we have to do an extra
                    # device code link step with a dummy link file
                    linko_path = os.path.join(cache_directory, 'link.o')
                    subprocess.check_output([nvcc_path, '-dlink', '-Xcompiler', '-fPIC',
                                     '-o', linko_path, o_path],
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True)
                    subprocess.check_output(['g++', '-shared',
                                     '-o', testlib_path, o_path, linko_path,
                                     '-lcuda'],
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True)
                except subprocess.CalledProcessError as exception:
                    tf.logging.log(tf.logging.ERROR, 'nvcc error: ' + exception.output)
                    raise

                # clean up .o files
                subprocess.call(['rm', o_path, linko_path])

                libtest = ctypes.cdll.LoadLibrary(testlib_path)

            self._test_cuda_op = libtest.testCUDAOperator
            self._test_cuda_op.restype = ctypes.c_int16
            self._test_cuda_op.argtypes = \
                [ctypes.c_char_p, ctypes.c_char_p,
                 ndpointer(dtype=_TensorParam, flags="C_CONTIGUOUS"), ctypes.c_size_t,
                 ndpointer(dtype=_TensorParam, flags="C_CONTIGUOUS"), ctypes.c_size_t,
                 ctypes.c_uint16,
                 ndpointer(dtype=ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t]

        err = self._test_cuda_op(lib_path, fcn_name,
                                 self._input_params, ctypes.c_size_t(num_inputs),
                                 self._output_params, ctypes.c_size_t(num_outputs),
                                 ctypes.c_uint16(cuda_threads_per_block),
                                 eval_times_ms, ctypes.c_size_t(iters))

        if err != 0 or np.isnan(eval_times_ms).any():
            tf.logging.log(tf.logging.ERROR, 'Test CUDA operator failed for Op ' + self.__class__.__name__)
            raise ValueError('Test CUDA operator failed for Op ' + self.__class__.__name__)

        if profiling_iterations is None:
            return Operator._unwrap_single(self._output_buffers)
        else:
            return Operator._unwrap_single(self._output_buffers), eval_times_ms

    # TODO - need to figure out how to test gradients
    # def evaluate_c_grad(self, *grads):
    #     """
    #     Evaluate the compiled C code for the gradient of this operation, mainly used for testing. This function uses
    #     a standalone runtime for compiling and running the operation so it does not depend on an external execution
    #     runtime. This also means that this function only works for operators whose inputs are numpy arrays.
    #
    #     :return: The resulting numpy array, or list of numpy arrays if there are multiple inputs.
    #     """
    #     for grad, tipe in zip(grads, self.output_types):
    #         if TensorType.like(grad) != tipe:
    #             raise TypeError('Invalid gradient tensor type')
    #
    #     if self._grad_c_function is None:
    #         c_lib = Operator._make_standalone_c(self.grad_c_src, self.grad_name)
    #         self._grad_c_function = getattr(c_lib, self.grad_name)
    #         self._grad_c_function.restype = ctypes.c_uint16
    #         self._grad_c_function.argtypes = self.grad_argtypes
    #
    #     self._allocate_grad_buffers()
    #     self._grad_c_function(*(self._inputs + list(grads) + self._grad_output_buffers))
    #     return Operator._unwrap_single(self._grad_output_buffers)

    # def evaluate_cuda_grad(self, *grads, **cuda_threads_per_block_kw):
    #     """
    #     Evaluate the compiled CUDA code for the gradient of this operation, generally only used for testing. This
    #     function uses a standalone runtime for compiling and running the operation so it does not depend
    #     on an external execution runtime. This also means that this function only works for operators whose
    #     inputs are numpy arrays. Note that a new CUDA context is used every time this is evaluated, causing
    #     execution times to be very slow.
    #
    #     :return: The resulting numpy array, or list of numpy arrays if there are multiple inputs.
    #     """
    #     for grad, tipe in zip(grads, self.output_types):
    #         if TensorType.like(grad) != tipe:
    #             raise TypeError('Invalid gradient tensor type')
    #
    #     if 'cuda_threads_per_block' in cuda_threads_per_block_kw.keys():
    #         cuda_threads_per_block = cuda_threads_per_block_kw['cuda_threads_per_block']
    #     else:
    #         cuda_threads_per_block = Operator._default_cuda_threads_per_block
    #
    #     if self._grad_cuda_function is None:
    #         c_lib = Operator._make_standalone_cuda(self.grad_cuda_src, self.grad_cuda_launch_template, self.grad_name)
    #         self._grad_cuda_function = getattr(c_lib, self.grad_name)
    #         self._grad_cuda_function.restype = ctypes.c_uint16
    #         self._grad_cuda_function.argtypes = self.grad_argtypes
    #
    #     self._allocate_grad_buffers()
    #     self._grad_cuda_function(*(self._inputs + list(grads) + self._grad_output_buffers + [cuda_threads_per_block]))
    #     return Operator._unwrap_single(self._grad_output_buffers)

    def as_tensorflow(self, cuda_threads_per_block=_default_cuda_threads_per_block):
        """
        Create a TensorFlow operator based on this operation and register it with the current TensorFlow Graph. The
        inputs to the operator must be numpy arrays or TensorFlow tensors. The operation will be evaluated later
        by the TensorFlow session.

        :param cuda_threads_per_block: number of cuda threads to use per thread block

        :return: A TensorFlow operator.
        """
        tf.logging.log(tf.logging.DEBUG, 'Compiling generic C++ for Op ' + self.__class__.__name__)
        cpu_op_lib = Operator._make_generic_c(self.op_c_generic, self.op_name)
        if cuda_enabled:
            tf.logging.log(tf.logging.DEBUG, 'Compiling generic CUDA for Op ' + self.__class__.__name__)
            cuda_op_lib = Operator._make_generic_cuda(self.op_cuda_generic, self.op_name)
        else:
            cuda_op_lib = ''

        if self.grad_name is None:
            gpu_grad_name = ''
            gpu_grad_lib = ''
            cpu_grad_name = ''
            cpu_grad_lib = ''
        else:
            tf.logging.log(tf.logging.DEBUG, 'Compiling generic C++ for gradient of Op ' + self.__class__.__name__)
            cpu_grad_lib = Operator._make_generic_c(self.grad_c_generic, self.grad_name)
            cpu_grad_name = self.grad_name + '_generic_cpp'
            if cuda_enabled:
                tf.logging.log(tf.logging.DEBUG, 'Compiling generic CUDA for gradient of Op ' + self.__class__.__name__)
                gpu_grad_lib = Operator._make_generic_cuda(self.grad_cuda_generic, self.grad_name)
                gpu_grad_name = self.grad_name + '_generic_cuda'
            else:
                gpu_grad_name = ''
                gpu_grad_lib = ''

        out_shapes = []
        out_types = []
        for cur_type in self.output_types:
            if cur_type.dtype == float32:
                tf_type = 'float'
            elif cur_type.dtype == float64:
                tf_type = 'double'
            else:
                raise NotImplementedError('Only floats and doubles currently supported.')

            out_types.append(tf_type)
            out_shapes.append(cur_type.shape)

        Operator._register_shape_inference()
        Operator._load_dynamiclib_module()
        Operator._register_gradient()
        tf_op = Operator._dynamiclibop_module.dynamic_lib(inputs=self._inputs,
                                                          out_shapes=out_shapes,
                                                          out_types=out_types,
                                                          cpu_lib_path=cpu_op_lib,
                                                          cpu_func_name=self.op_name + '_generic_cpp',
                                                          gpu_lib_path=cuda_op_lib,
                                                          gpu_func_name=self.op_name + '_generic_cuda',
                                                          gpu_grad_func_name=gpu_grad_name,
                                                          gpu_grad_lib_path=gpu_grad_lib,
                                                          cpu_grad_func_name=cpu_grad_name,
                                                          cpu_grad_lib_path=cpu_grad_lib,
                                                          cuda_threads_per_block=cuda_threads_per_block)
        if len(out_shapes) == 1:
            return tf_op[0]
        else:
            return tf_op
