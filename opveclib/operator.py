
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
import hashlib
import os
import inspect
import subprocess
import string
import re

import tensorflow as tf
import numpy as np
from numpy.ctypeslib import ndpointer

from .expression import TensorType, ExpressionDAG, input, float32, float64, OutputTensor
from .local import version, cache_directory, cuda_enabled, cuda_directory
from . import language_pb2 as lang

_default_cuda_threads_per_block = 32


class _DynamicLibOp(object):
    _loaded_module = None
    _shape_infernence_registered = False

    @staticmethod
    def module():
        if _DynamicLibOp._loaded_module is None:
            libname = 'dynamiclibop.so.' + version
            dynamiclibop_path = os.path.join(cache_directory, libname)

            # build the library if it does not exist already
            if not os.path.exists(dynamiclibop_path):
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

            _DynamicLibOp._loaded_module = tf.load_op_library(dynamiclibop_path)

        return _DynamicLibOp._loaded_module

    @staticmethod
    def register_shape_inference():
        if _DynamicLibOp._shape_infernence_registered is False:
            _DynamicLibOp._shape_infernence_registered = True

            @tf.RegisterShape("DynamicLib")
            def get_out_shapes(op):
                return op.get_attr('out_shapes')


class _TensorParam(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p),
                ("dtype", ctypes.c_int),
                ("len", ctypes.c_size_t)]


class UndefinedGradientError(NotImplementedError):
    pass


class _OperatorOutput(object):
    """
    Class which represents an un-evaluated output tensor, used for building lazily evaluated DAGs of operators
    """
    def __init__(self, parent, index):
        if not isinstance(parent, Operator):
            raise TypeError('parent must be an Operator')
        if not isinstance(index, int):
            raise TypeError('index must be an int')

        self.parent = parent
        self.index = index
        self.shape = parent.output_types[index].shape
        self.dtype = parent.output_types[index].dtype


def _resolve_output(x):
    """
    Resolve whether or not an object is an _OperatorOutput. Converts single-output Operators to an _OperatorOutput

    :param x: the argument to resolve
    :return: the argument converted into an _OperatorOutput
    :raises ValueError: When argument is a multi-output Operator
    :raises TypeError: When argument is neither an Operator nor an _OperatorOutput
    """
    if isinstance(x, Operator):
        if len(x.output_types) is not 1:
            raise ValueError('Only a single-output Operator can be used as an input to another operator. '
                             'Index a specific output from multi-output Operators.')
        return x[0]
    elif not isinstance(x, _OperatorOutput):
        raise TypeError('Only operator outputs can be used to build an op dag.')
    else:
        return x


def _op_hash(op):
    return 'f' + hashlib.sha224(op.SerializeToString() + version.encode('utf-8')).hexdigest()


class Operator(object):
    """
    Class which is extended to define a new operator and its gradient.
    """

    # @staticmethod
    # def _register_gradient():
    #     if not Operator._gradient_registered:
    #         Operator._gradient_registered = True
    #
    #         from tensorflow.python.framework import ops as tf_ops
    #
    #         @tf_ops.RegisterGradient("DynamicLib")
    #         def _dynamic_lib_grad(op, *grads_above):
    #             num_inputs = len(op.inputs)
    #
    #             gpu_grad_name = op.get_attr('gpu_grad_func_name')
    #             gpu_grad_lib = op.get_attr('gpu_grad_lib_path')
    #             cpu_grad_name = op.get_attr('cpu_grad_func_name')
    #             cpu_grad_lib = op.get_attr('cpu_grad_lib_path')
    #             cuda_threads_per_block = op.get_attr('cuda_threads_per_block')
    #
    #             if cpu_grad_name == '':
    #                 grads = []
    #                 for i in range(num_inputs):
    #                     grads.append(None)
    #
    #                 return grads
    #             else:
    #                 out_shapes = []
    #                 out_types = []
    #                 for cur_input in op.inputs:
    #                     cur_type = TensorType.like(cur_input)
    #                     if cur_type.dtype == float32:
    #                         tf_type = 'float'
    #                     elif cur_type.dtype == float64:
    #                         tf_type = 'double'
    #                     else:
    #                         raise NotImplementedError('Only floats and doubles currently supported.')
    #
    #                     out_types.append(tf_type)
    #                     out_shapes.append(cur_type.shape)
    #
    #                 inputs = []
    #                 for inp in op.inputs:
    #                     inputs.append(inp)
    #
    #                 try:
    #                     len(grads_above)
    #                 except TypeError:
    #                     inputs.append(grads_above)
    #                 else:
    #                     for grad_above in list(grads_above):
    #                         inputs.append(grad_above)
    #
    #                 grads = Operator._dynamiclibop_module.dynamic_lib(inputs=inputs,
    #                                                                   out_shapes=out_shapes,
    #                                                                   out_types=out_types,
    #                                                                   cpu_lib_path=cpu_grad_lib,
    #                                                                   cpu_func_name=cpu_grad_name,
    #                                                                   gpu_lib_path=gpu_grad_lib,
    #                                                                   gpu_func_name=gpu_grad_name,
    #                                                                   gpu_grad_func_name='',
    #                                                                   gpu_grad_lib_path='',
    #                                                                   cpu_grad_func_name='',
    #                                                                   cpu_grad_lib_path='',
    #                                                                   cuda_threads_per_block=cuda_threads_per_block)
    #                 return grads

    @staticmethod
    def _unwrap_single(x):
        if len(x) is 1:
            return x[0]
        else:
            return x

    def __init__(self, *inputs, **kwargs):
        # set default options
        self._options = kwargs

        tf.logging.log(tf.logging.DEBUG, 'Creating Op ' + self.__class__.__name__)

        self._inputs = list(inputs)

        self._input_types = []
        for inp_n, inp in enumerate(inputs):
            try:
                inp = _resolve_output(inp)
            except TypeError:
                pass

            try:
                self._input_types.append(TensorType.like(inp))
            except AttributeError:
                raise TypeError('Received a ' + inp.__class__.__name__ + ' instead of a tensor at argument position ' +
                                str(inp_n + 1) + ' in the Op constructor. ' +
                                'Should this argument be passed as a constant (keyword argument) instead?')

        num_inputs = len(self._input_types)

        # parse arg spec of the function and build up a dictionary of defaults to be applied if constants
        # of the same name are not passed to contructor
        arg_spec = inspect.getargspec(self.op).args[1:]
        defaults = inspect.getargspec(self.op).defaults
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
                err_msg += 'op function received ' + str(len(constant_names)) + ' constants:\n' + \
                           str(constant_names) + '\n'
                additional = ' additional'

            err_msg += 'Based on constructor call pattern, op function signature expects ' + \
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
        for cur_type in self._input_types:
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
        output_exprs = self.op(*args)
        if output_exprs is None:
            raise ValueError('No outputs returned from op function')

        # wrap as list if only one output
        try:
            len(output_exprs)
        except TypeError:
            output_exprs = [output_exprs]

        # make sure number of returned parameters equals the number of declared outputs
        if len(output_exprs) != ExpressionDAG.num_outputs:
            raise ValueError('Defined ' + str(ExpressionDAG.num_outputs) + ' outputs, but returned ' +
                             str(len(output_exprs)) +
                             '. Number of defined outputs must equal number of returned outputs.')

        # make sure all returned values are output expressions
        # reorder output io_index according to return order instead of declaration order
        self.output_types = []
        prev_index = []
        for index, expr in enumerate(output_exprs):
            if type(expr) is not OutputTensor:
                raise TypeError('User functions must only return outputs. Instead got:\n' + str(expr))
            prev_index.append(ExpressionDAG.expr_index(expr))
            expr.proto_expr.io_index = index
            self.output_types.append(TensorType.like(expr))
        # reorder declaration of outputs in expression dag
        prev_index.sort()
        for index, expr in zip(prev_index, output_exprs):
            ExpressionDAG.exprs[index] = expr
            ExpressionDAG.expr_ids[index] = id(expr)

        self.expression_dag = ExpressionDAG.as_proto()
        ExpressionDAG.clear()

    def __getitem__(self, item):
        return _OperatorOutput(self, item)

    def op(self, *input_tensors, **constants):
        """
        Abstract member that must be implemented to define an operator

        :param input_tensors: tensor arguments
        :param constants: constant arguments, passed in by keyword
        :return: Must return a list of output tensors, defined by this function
        """
        raise NotImplementedError("Abstract class")

    def grad(self, *input_tensors, **constants):
        """
        Abstract member that must be implemented to define an operator's gradient function

        :param input_tensors: tensor arguments which must be in the same order and same type as the inputs and outputs
            of the op function
        :param constants: constant arguments, passed in by keyword
        :return: Must return a list of output tensors, equal in TensorType to the inputs of the op function
        """
        raise UndefinedGradientError()


def _build_op_dag(*outputs):
    """
    Perform BFS on the op nodes
    :param outputs: a list of the operator outputs from which to build the dag
    :return: a tuple containing the op DAG protobuf and a list of the tensor inputs to the DAG
    """

    ops = []
    op_ids = []
    op_depth = []
    input_indices = []
    dag_inputs = []
    dag_input_ids = []

    def traverse(cur_node):
        if not isinstance(cur_node, Operator):
            raise TypeError()

        cur_id = id(cur_node)

        # add unvisited ops (nodes) to the op list
        if cur_id not in op_ids:
            op_ids.append(cur_id)
            ops.append(cur_node)
            op_depth.append(None)
            input_indices.append(None)
            cur_index = len(op_ids) - 1

            # tabulate each input tensor (edge) for this op. visit parent ops if inputs come from other ops.
            cur_input_indices = []
            max_depth = -1
            for cur_input in cur_node._inputs:
                try:
                    resolved = _resolve_output(cur_input)
                    parent = resolved.parent
                    parent_id, parent_depth = traverse(parent)
                    max_depth = max(max_depth, parent_depth)
                    parent_index = op_ids.index(parent_id)
                    output_index = resolved.index
                    dag_input_index = None
                except TypeError:
                    if id(cur_input) not in dag_input_ids:
                        dag_inputs.append(cur_input)
                        dag_input_ids.append(id(cur_input))
                    parent_index = None
                    output_index = None
                    dag_input_index = dag_input_ids.index(id(cur_input))

                cur_input_indices.append({'parent_index': parent_index,
                                          'output_index': output_index,
                                          'dag_input_index': dag_input_index})

            input_indices[cur_index] = cur_input_indices
            cur_depth = max_depth + 1
            op_depth[cur_index] = cur_depth
        else:
            cur_index = op_ids.index(cur_id)
            cur_depth = op_depth[cur_index]

        return cur_id, cur_depth

    output_indices = []
    for dag_output in outputs:
        cur_output = _resolve_output(dag_output)
        parent_id, parent_depth = traverse(cur_output.parent)
        parent_index = op_ids.index(parent_id)
        output_index = cur_output.index
        output_indices.append({'parent_index': parent_index, 'output_index': output_index})

    # sort ops according to their DAG depth
    new_to_old = np.argsort(op_depth)
    old_to_new = np.argsort(new_to_old)
    sorted_ops = []
    sorted_input_indices = []
    for new_index, old_index in enumerate(new_to_old):
        sorted_ops.append(ops[old_index])
        sorted_input_indices.append([])
        cur_input_indices = sorted_input_indices[-1]
        for cur_index in input_indices[old_index]:
            parent_index = cur_index['parent_index']
            output_index = cur_index['output_index']
            dag_input_index = cur_index['dag_input_index']
            if parent_index is not None:
                parent_index = old_to_new[parent_index]
            cur_input_indices.append({'parent_index': parent_index,
                                      'output_index': output_index,
                                      'dag_input_index': dag_input_index})

    # update parent indices for the outputs
    for output_index in output_indices:
        output_index['parent_index'] = old_to_new[output_index['parent_index']]

    # create the protobuf representation
    op_dag = lang.OperatorDAG()
    for op in sorted_ops:
        op_dag.operators.add().CopyFrom(op.expression_dag)

    for dag_input in dag_inputs:
        proto_type = TensorType.like(dag_input).as_proto()
        op_dag.dag_input_types.add().CopyFrom(proto_type)

    for input_index in sorted_input_indices:
        ref_list = lang.OperatorDAG.OperatorInputReferences()
        for cur_ref in input_index:
            proto_ref = lang.OperatorDAG.OperatorInputReference()
            if cur_ref['parent_index'] is None:
                proto_ref.is_leaf = True
                proto_ref.dag_input_index = cur_ref['dag_input_index']
            else:
                proto_ref.op_index = cur_ref['parent_index']
                proto_ref.op_output_index = cur_ref['output_index']

            ref_list.input_refs.add().CopyFrom(proto_ref)

        op_dag.references.add().CopyFrom(ref_list)

    for output_index in output_indices:
        proto_ref = lang.OperatorDAG.DAGOutputReference()
        proto_ref.op_index = output_index['parent_index']
        proto_ref.op_output_index = output_index['output_index']
        op_dag.dag_outputs.add().CopyFrom(proto_ref)

    return op_dag, dag_inputs


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


def evaluate(output_list, profiling_iterations=1, target_language='cpp'):
    """
    Evaluate a collection of OVL operator, mainly used for testing. This function uses a test operator
    function for running the generated generic version of the operator so it does not depend on an external
    execution runtime. This also means that this function only works for operators whose inputs are numpy arrays.

    :param output_list: The outputs to evaluate
    :param profiling_iterations: Number of times to run this operator for profiling purposes.
        Must be a positive int.
    :param target_language: 'cpp' or 'cuda'

    :return:  If profiling_iterations is set to None, returns the numpy array, or list of numpy arrays if there are
        multiple outputs, containing results from evaluation. If profiling_iterations is set, returns a tuple of the
        output array(s), and a numpy array that contains the time, in ms, that each function evaluation took.
    """

    # Generate the protobuf header file.
    # Since all we need for the test libraries is the DType enum, do not use protoc to generate the
    # fully functional protobuf code, since this introduces a dependency on the C++ protobuf development libraries.
    proto_header = os.path.join(cache_directory, 'language_dtype.h')
    if not os.path.exists(proto_header):
        enum_src = ''
        enum_val = 0
        for enum_name in lang._DTYPE.values:
            enum_src += '    ' + enum_name.name + ' = ' + str(enum_val) + ',\n'
            enum_val += 1

        # generate header enum
        h_src = """
        |//Generated Code - do not edit
        |#ifndef LANGUAGE_DTYPE_H
        |#define LANGUAGE_DTYPE_H
        |namespace opveclib {
        |enum DType {
        |${enum_src}
        |};
        |}
        |#endif  // LANGUAGE_DTYPE_H
        """
        h_src = string.Template(h_src).substitute(locals())
        h_src = re.sub('\n[ \t]*\|', '\n', h_src)
        with open(proto_header, 'w') as f:
            f.write(h_src)

    # Dynamically load the test library, compile if necessary
    invalid_language = 'Unsupported target_language: ' + target_language
    if target_language == 'cpp':
        testlib_path = os.path.join(cache_directory, 'libtestcop.so.'+version)
        try:
            libtest = ctypes.cdll.LoadLibrary(testlib_path)
        except OSError:
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

        test_c_op = libtest.testCOperator
        test_c_op.restype = ctypes.c_int16
        test_c_op.argtypes = \
            [ctypes.c_char_p, ctypes.c_char_p,
             ndpointer(dtype=_TensorParam, flags="C_CONTIGUOUS"), ctypes.c_size_t,
             ndpointer(dtype=_TensorParam, flags="C_CONTIGUOUS"), ctypes.c_size_t,
             ndpointer(dtype=ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t]

    elif target_language == 'cuda':
        testlib_path = os.path.join(cache_directory, 'libtestcudaop.so.'+version)
        try:
            libtest = ctypes.cdll.LoadLibrary(testlib_path)
        except OSError:
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

        test_cuda_op = libtest.testCUDAOperator
        test_cuda_op.restype = ctypes.c_int16
        test_cuda_op.argtypes = \
            [ctypes.c_char_p, ctypes.c_char_p,
             ndpointer(dtype=_TensorParam, flags="C_CONTIGUOUS"), ctypes.c_size_t,
             ndpointer(dtype=_TensorParam, flags="C_CONTIGUOUS"), ctypes.c_size_t,
             ctypes.c_uint16,
             ndpointer(dtype=ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t]
    else:
        raise ValueError(invalid_language)

    dag, inputs = _build_op_dag(*output_list)

    output_buffers = []
    # compile all ops in the dag
    for op_index, op in enumerate(dag.operators):
        name = _op_hash(op)

        # generate code
        op_c_src, op_cuda_src, op_cuda_launch_template, op_c_generic, op_cuda_generic = \
            ExpressionDAG.generate(op, name)

        input_types, output_types = ExpressionDAG.io_types()
        num_inputs = len(input_types)
        num_outputs = len(output_types)

        eval_times_ms = np.empty(profiling_iterations, dtype=np.float64)
        eval_times_ms[:] = np.nan

        cur_input_params = np.empty(num_inputs, dtype=_TensorParam)
        for input_index, input_ref in enumerate(dag.references[op_index].input_refs):
            if input_ref.is_leaf:
                cur_buffer = inputs[input_ref.dag_input_index]
            else:
                cur_buffer = output_buffers[input_ref.op_index][input_ref.op_output_index]

            cur_data = cur_buffer.ctypes.data
            cur_dtype = ctypes.c_int(input_types[input_index].dtype.proto_dtype)
            cur_len = ctypes.c_size_t(input_types[input_index].size)
            cur_input_params[input_index] = _TensorParam(data=cur_data, dtype=cur_dtype, len=cur_len)

        # allocate output memory for the current operator
        cur_output_params = np.empty(num_outputs, dtype=_TensorParam)
        output_buffers.append([])
        cur_buffers = output_buffers[-1]
        for output_index, output_type in enumerate(output_types):
            t = output_type.dtype.as_numpy()
            new_buffer = np.empty(output_type.shape, dtype=t)
            cur_buffers.append(new_buffer)

            cur_data = new_buffer.ctypes.data
            cur_dtype = ctypes.c_int(output_type.dtype.proto_dtype)
            cur_len = ctypes.c_size_t(output_type.size)
            cur_output_params[output_index] = _TensorParam(data=cur_data, dtype=cur_dtype, len=cur_len)

        # evaluate the outputs
        if target_language == 'cpp':
            lib_path = _make_generic_c(op_c_generic, name)
            err = test_c_op(lib_path, name+'_generic_cpp',
                            cur_input_params, ctypes.c_size_t(num_inputs),
                            cur_output_params, ctypes.c_size_t(num_outputs),
                            eval_times_ms,
                            ctypes.c_size_t(profiling_iterations))
        elif target_language == 'cuda':
            lib_path = _make_generic_cuda(op_cuda_generic, name)
            # TODO: expose this parameter to the user?

            err = test_cuda_op(lib_path, name+'_generic_cuda',
                               cur_input_params, ctypes.c_size_t(num_inputs),
                               cur_output_params, ctypes.c_size_t(num_outputs),
                               ctypes.c_uint16(_default_cuda_threads_per_block),
                               eval_times_ms,
                               ctypes.c_size_t(profiling_iterations))
        else:
            raise ValueError(invalid_language)

        # TODO: deallocate output buffers that are no longer needed

    outputs = []
    for out_ref in dag.dag_outputs:
        outputs.append(output_buffers[out_ref.op_index][out_ref.op_output_index])

    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


def as_tensorflow(tensor_list):
    """
    Create a DAG of TensorFlow operators based on a DAG OVL operators and register it with the current
    TensorFlow Graph. The inputs to the DAG must be numpy arrays or TensorFlow tensors.

    :param tensors: output tensors to

    :return: A TensorFlow operator.
    """
    dag, inputs = _build_op_dag(*tensor_list)

    output_tensors = []
    # compile all ops in the dag
    for op_index, op in enumerate(dag.operators):
        name = _op_hash(op)

        # generate code
        op_c_src, op_cuda_src, op_cuda_launch_template, op_c_generic, op_cuda_generic = \
            ExpressionDAG.generate(op, name)

        tf.logging.log(tf.logging.DEBUG, 'Compiling generic C++ for Op ' + name)
        cpu_op_lib = _make_generic_c(op_c_generic, name)
        if cuda_enabled:
            tf.logging.log(tf.logging.DEBUG, 'Compiling generic CUDA for Op ' + name)
            cuda_op_lib = _make_generic_cuda(op_cuda_generic, name)
        else:
            cuda_op_lib = ''

        input_types, output_types = ExpressionDAG.io_types()

        out_shapes = []
        out_tf_types = []
        for cur_type in output_types:
            out_tf_types.append(cur_type.dtype.as_tensorflow())
            out_shapes.append(cur_type.shape)

        cur_inputs = []
        for ref in dag.references[op_index].input_refs:
            if ref.is_leaf:
                cur_inputs.append(inputs[ref.dag_input_index])
            else:
                cur_inputs.append(output_tensors[ref.op_index][ref.op_output_index])

        tf_op = _DynamicLibOp.module().dynamic_lib(inputs=cur_inputs,
                                                   out_shapes=out_shapes,
                                                   out_types=out_tf_types,
                                                   cpu_lib_path=cpu_op_lib,
                                                   cpu_func_name=name + '_generic_cpp',
                                                   gpu_lib_path=cuda_op_lib,
                                                   gpu_func_name=name + '_generic_cuda',
                                                   gpu_grad_func_name='',
                                                   gpu_grad_lib_path='',
                                                   cpu_grad_func_name='',
                                                   cpu_grad_lib_path='',
                                                   cuda_threads_per_block=_default_cuda_threads_per_block)
        output_tensors.append(tf_op)

    outputs = []
    for out_ref in dag.dag_outputs:
        outputs.append(output_tensors[out_ref.op_index][out_ref.op_output_index])

    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs
