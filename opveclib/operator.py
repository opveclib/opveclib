
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
from collections import namedtuple
import numpy as np
from numpy.ctypeslib import ndpointer

from .expression import TensorType, ExpressionDAG, input, OutputTensor
from .local import version, cache_directory, cuda_enabled, cuda_directory, logger
from . import language_pb2 as lang

_default_cuda_threads_per_block = 32


class _DynamicLibOp(object):
    _loaded_module = None
    _shape_infernence_registered = False
    _gradient_registered = False

    @staticmethod
    def module():
        import tensorflow as tf
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
                        logger.debug('*** building dynamiclibop for GPU')
                        subprocess.check_output(['g++', '-fPIC', '-Wall', '-shared',
                                                 '-std=c++11', '-O2', '-Wextra', '-DGOOGLE_CUDA=1',
                                                 '-o', dynamiclibop_path,
                                                 this_directory + '/dynamiclibop.cc',
                                                 '-isystem', cuda_directory + '/include',
                                                 '-isystem', tf_include],
                                                stderr=subprocess.STDOUT,
                                                universal_newlines=True)
                    else:
                        logger.debug('*** building dynamiclibop for CPU')
                        subprocess.check_output(['g++', '-fPIC', '-Wall', '-shared',
                                                 '-std=c++11', '-O2', '-Wextra',
                                                 '-o', dynamiclibop_path,
                                                 this_directory + '/dynamiclibop.cc',
                                                 '-isystem', tf_include],
                                                stderr=subprocess.STDOUT,
                                                universal_newlines=True)
                except subprocess.CalledProcessError as exception:
                    logger.debug('g++ error: ' + exception.output)
                    raise

            _DynamicLibOp._loaded_module = tf.load_op_library(dynamiclibop_path)

        if _DynamicLibOp._shape_infernence_registered is False:
            _DynamicLibOp._shape_infernence_registered = True

            @tf.RegisterShape("DynamicLib")
            def get_out_shapes(op):
                return op.get_attr('out_shapes')

        if not _DynamicLibOp._gradient_registered:
            from tensorflow.python.framework import ops as tf_ops

            @tf_ops.RegisterGradient("DynamicLib")
            def _dynamic_lib_grad(op, *grads):
                if op.get_attr('serialized_grad_dag') == '':
                    return [None]*len(op.inputs)

                grad_dag = lang.OperatorDAG()
                grad_dag.ParseFromString(op.get_attr('serialized_grad_dag'))

                try:
                    len(grads)
                except TypeError:
                    grad_list = [grads]
                else:
                    grad_list = list(grads)

                grad_inputs = []
                grad_of_grad_dags = []
                for op_input in op.inputs:
                    grad_inputs.append(op_input)
                    grad_of_grad_dags.append(None)
                for grad in grad_list:
                    grad_inputs.append(grad)
                    grad_of_grad_dags.append(None)

                # make sure that the input types and expected types are consistent
                for grad_input_index, grad_input in enumerate(grad_inputs):
                    received_type = TensorType.like(grad_input).as_proto()
                    expected_type = grad_dag.dag_input_types[grad_input_index]

                    if received_type != expected_type:
                        raise TypeError('Received a tensor of type: ' + str(received_type) +
                                        ', but expected a type: ' + str(expected_type) +
                                        ' at gradient input index: ' + str(grad_input_index))

                return _dag_to_tf(grad_dag, grad_inputs, grad_of_grad_dags)

            _DynamicLibOp._gradient_registered = True

        return _DynamicLibOp._loaded_module


class _TensorParam(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p),
                ("dtype", ctypes.c_int),
                ("len", ctypes.c_size_t)]


class _OperatorOutput(object):
    """
    Class which represents an un-evaluated output tensor, used for building lazily evaluated DAGs of operators
    """
    def __init__(self, parent, index):
        if not isinstance(parent, _Operator):
            raise TypeError('parent must be an Operator')
        if not isinstance(index, int):
            raise TypeError('index must be an int')

        self.parent = parent
        self.index = index
        self.shape = parent.output_types[index].shape
        self.dtype = parent.output_types[index].dtype


class _GradientPlaceholder(object):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _Operator(object):
    """
    Class which is extended to define a new operator and its gradient.
    """
    def __init__(self, dag, output_types, inputs, grad_dag, name):
        self.inputs = inputs
        self.expression_dag = dag
        self.output_types = output_types
        self.name = name

        self.grad_dag = grad_dag

    def __getitem__(self, item):
        return _OperatorOutput(self, item)


def _resolve_output(x):
    """
    Resolve whether or not an object is an _OperatorOutput. Converts single-output Operators to an _OperatorOutput

    :param x: the argument to resolve
    :return: the argument converted into an _OperatorOutput
    :raises ValueError: When argument is a multi-output Operator
    :raises TypeError: When argument is neither an Operator nor an _OperatorOutput
    """
    if isinstance(x, _Operator):
        if len(x.output_types) is not 1:
            raise ValueError('Only a single-output Operator can be used as an input to another operator. '
                             'Index a specific output from multi-output Operators.')
        return x[0]
    elif not isinstance(x, _OperatorOutput):
        raise TypeError('Only operator outputs can be used to build an op dag. Received a ' + str(type(x)))
    else:
        return x


def _op_hash(op):
    return 'f' + hashlib.sha224(op.SerializeToString() + version.encode('utf-8')).hexdigest()


class _OpGenerator(object):
    def __init__(self, op_function, forbid_none_valued_constants):
        self.op_function = op_function
        self.forbid_none_valued_constants = forbid_none_valued_constants

        self.grad_function = None

    def __str__(self):
        return self.op_function.__name__

    def __call__(self, *inputs, **defined_constants):
        func_args, func_varargs, func_keywords, func_defaults = inspect.getargspec(self.op_function)
        if func_defaults is None:
            input_names = func_args
            constants = {}
        else:
            input_names = func_args[:-len(func_defaults)]
            constants = dict(zip(func_args[-len(func_defaults):], func_defaults))
            constants.update(defined_constants)

        f_name = self.op_function.__name__

        if self.forbid_none_valued_constants:
            for key in constants.keys():
                if constants[key] is None:
                    raise ValueError(f_name + ' argument ' + key + ' is None, which implies an unset constant.\n'
                                     '  If a None constant is meaningful for this operator, the operator should be '
                                     'defined with the appropriate decorator flag.')

        input_types = []
        for inp_n, inp in enumerate(inputs):
            try:
                inp = _resolve_output(inp)
            except TypeError:
                pass

            try:
                input_types.append(TensorType.like(inp))
            except AttributeError:
                raise TypeError('Unexpectedly received a ' + inp.__class__.__name__ +
                                ' instead of a tensor at argument position ' +
                                str(inp_n + 1) + ' in the call to ' + f_name + '.  '
                                'Should this argument be passed as a constant (keyword argument) instead?')

        num_inputs = len(input_types)

        if len(input_names) != num_inputs:
            err_msg = '\n'
            err_msg += f_name + ' function signature expects ' + str(len(input_names)) + \
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

        # interpret function to build up ExpressionDAG
        output_exprs = self.op_function(*input_exprs, **constants)

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
        expression_dag.name = f_name

        if self.grad_function is None:
            grad_dag = None
        else:
            grad_args, grad_varargs, grad_keywords, grad_defaults = inspect.getargspec(self.grad_function)
            if len(output_types) + len(input_names) != len(grad_args[:-len(grad_defaults)]):
                raise SyntaxError('Gradient function must have inputs equal to the sum of the number of '
                                  'inputs and outputs of the operator function.')

            grad_inputs = []
            for t in input_types:
                grad_inputs.append(_GradientPlaceholder(t.shape, t.dtype))
            for t in output_types:
                grad_inputs.append(_GradientPlaceholder(t.shape, t.dtype))

            # interpret gradient function
            grad_outputs = self.grad_function(*grad_inputs, **constants)

            # wrap as list if only one output
            try:
                len(grad_outputs)
            except TypeError:
                grad_outputs = [grad_outputs]

            # make sure grad outputs are the same type as op inputs
            for grad_output_index, grad_output in enumerate(grad_outputs):
                cur_input_type = input_types[grad_output_index]
                cur_grad_output_type = TensorType.like(_resolve_output(grad_output))
                if cur_input_type != cur_grad_output_type:
                    raise TypeError('Gradient output index ' + str(grad_output_index) + ', with TensorType: ' +
                                    str(cur_grad_output_type) + ' is inconsistent with operator input index ' +
                                    str(grad_output_index) + ', with TensorType: ' + str(cur_input_type))

            grad_dag = _build_op_dag(*grad_outputs).proto_dag

        return _Operator(expression_dag, output_types, inputs, grad_dag, f_name)

    def add_grad(self, grad_function):
        if self.grad_function is None:
            self.grad_function = grad_function
        else:
            raise ValueError('Gradient function is already defined.')


def operator(forbid_none_valued_constants=True):
    def wrapper(op_function):
        if inspect.getargspec(op_function).keywords is not None:
            raise SyntaxError('Operator functions cannot accept keyword arguments without default values.')

        # TODO: implement vararg input parsing
        if inspect.getargspec(op_function).varargs is not None:
            raise NotImplementedError('Operator functions cannot accept varags. '
                                      'This functionality may be enabled in the future.')

        return _OpGenerator(op_function, forbid_none_valued_constants)
    return wrapper


def gradient(op_function):
    if not isinstance(op_function, _OpGenerator):
        raise TypeError('gradient decorator argument must be a function decorated as an operator')

    def wrapper(grad_function):
        func_args, func_varargs, func_keywords, func_defaults = inspect.getargspec(op_function.op_function)
        grad_args, grad_varargs, grad_keywords, grad_defaults = inspect.getargspec(grad_function)

        func_input_names = func_args[:-len(func_defaults)]
        func_constants = dict(zip(func_args[-len(func_defaults):], func_defaults))

        grad_input_names = grad_args[:-len(grad_defaults)]
        grad_constants = dict(zip(grad_args[-len(func_defaults):], grad_defaults))

        if func_constants != grad_constants:
            raise SyntaxError('Constant argument names and default values must be identical for '
                              'the op function and its gradient.')

        if func_input_names != grad_input_names[:len(func_input_names)]:
            raise SyntaxError('Gradient function must have same initial argument names as the op function.')

        op_function.add_grad(grad_function)

    return wrapper


_OperatorDAG = namedtuple('_OperatorDAG', ['proto_dag', 'inputs', 'operators', 'grad_dags'])


def _build_op_dag(*outputs):
    """
    Perform BFS on the op nodes
    :param outputs: a list of the operator outputs from which to build the dag
    :return: an _OpDAG
    """

    ops = []
    op_ids = []
    op_depth = []
    input_indices = []
    dag_inputs = []
    dag_input_ids = []

    def traverse(cur_node):
        if not isinstance(cur_node, _Operator):
            raise TypeError()

        cur_id = id(cur_node)

        # add unvisited ops (nodes) to the op list
        if cur_id not in op_ids:
            op_ids.append(cur_id)
            ops.append(cur_node)
            op_depth.append(None)
            input_indices.append(None)
            traverse_cur_index = len(op_ids) - 1

            # tabulate each input tensor (edge) for this op. visit parent ops if inputs come from other ops.
            traverse_cur_input_indices = []
            max_depth = -1
            for cur_input in cur_node.inputs:
                try:
                    resolved = _resolve_output(cur_input)
                    parent = resolved.parent
                    traverse_parent_id, traverse_parent_depth = traverse(parent)
                    max_depth = max(max_depth, traverse_parent_depth)
                    traverse_parent_index = op_ids.index(traverse_parent_id)
                    traverse_output_index = resolved.index
                    traverse_dag_input_index = None
                except TypeError:
                    if id(cur_input) not in dag_input_ids:
                        dag_inputs.append(cur_input)
                        dag_input_ids.append(id(cur_input))
                    traverse_parent_index = None
                    traverse_output_index = None
                    traverse_dag_input_index = dag_input_ids.index(id(cur_input))

                traverse_cur_input_indices.append({'parent_index': traverse_parent_index,
                                                   'output_index': traverse_output_index,
                                                   'dag_input_index': traverse_dag_input_index})

            input_indices[traverse_cur_index] = traverse_cur_input_indices
            cur_depth = max_depth + 1
            op_depth[traverse_cur_index] = cur_depth
        else:
            traverse_cur_index = op_ids.index(cur_id)
            cur_depth = op_depth[traverse_cur_index]

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
                proto_ref.op_index = int(cur_ref['parent_index'])
                proto_ref.op_output_index = int(cur_ref['output_index'])

            ref_list.input_refs.add().CopyFrom(proto_ref)

        op_dag.references.add().CopyFrom(ref_list)

    for output_index in output_indices:
        proto_ref = lang.OperatorDAG.DAGOutputReference()
        proto_ref.op_index = int(output_index['parent_index'])
        proto_ref.op_output_index = int(output_index['output_index'])
        op_dag.dag_outputs.add().CopyFrom(proto_ref)

    grad_dags = []
    for op in sorted_ops:
        grad_dags.append(op.grad_dag)

    dag = _OperatorDAG(proto_dag=op_dag, inputs=dag_inputs, operators=sorted_ops, grad_dags=grad_dags)
    return dag


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
            logger.debug('g++ error: ' + exception.output)
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
            logger.debug('nvcc error: ' + exception.output)
            raise

        # clean up .o files
        subprocess.call(['rm', generic_cuda_o_path])

    return generic_cuda_so_path


def evaluate(output_list, target_language='cpp'):
    """
    Evaluate a collection of OVL operator, mainly used for testing. This function uses a test operator
    function for running the generated generic version of the operator so it does not depend on an external
    execution runtime. This also means that this function only works for operators whose inputs are numpy arrays.

    :param output_list: The outputs to evaluate
    :param target_language: 'cpp' or 'cuda'

    :return:  A list of numpy arrays for each operator output in output_list
    """
    evaluated_outputs = profile(output_list, target_language=target_language, profiling_iterations=1)[0]

    if len(evaluated_outputs) == 1:
        return evaluated_outputs[0]
    else:
        return evaluated_outputs


def profile(output_list, target_language='cpp', profiling_iterations=1):
    """
    Evaluate a collection of OVL operator, mainly used for testing. This function uses a test operator
    function for running the generated generic version of the operator so it does not depend on an external
    execution runtime. This also means that this function only works for operators whose inputs are numpy arrays.

    :param output_list: The outputs to evaluate
    :param profiling_iterations: Number of times to run this operator for profiling purposes.
        Must be a positive int.
    :param target_language: 'cpp' or 'cuda'

    :return:  A tuple containing a list of numpy arrays for each operator output in output_list, and a dictionary of
        numpy arrays containing the execution times for each operator in the operator DAG.
    """

    # Generate the protobuf header file.
    # Since all we need for the test libraries is the DType enum, do not use protoc to generate the
    # fully functional protobuf code, since this introduces a dependency on the C++ protobuf development libraries.
    proto_header = os.path.join(cache_directory, 'language_dtype.h')
    if not os.path.exists(proto_header):
        enum_src = ''
        for enum_name, enum_val in lang.DType.items():
            enum_src += '    ' + enum_name + ' = ' + str(enum_val) + ',\n'

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
                logger.debug('g++ error: ' + exception.output)
                raise

            libtest = ctypes.cdll.LoadLibrary(testlib_path)

        test_c_op = libtest.testCOperator
        test_c_op.restype = ctypes.c_int16
        test_c_op.argtypes = \
            [ctypes.c_char_p, ctypes.c_char_p,
             ndpointer(dtype=_TensorParam, flags="C_CONTIGUOUS"), ctypes.c_size_t,
             ndpointer(dtype=_TensorParam, flags="C_CONTIGUOUS"), ctypes.c_size_t,
             ndpointer(dtype=ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t]
        test_cuda_op = None
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
                logger.debug('nvcc error: ' + exception.output)
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
        test_c_op = None
    else:
        raise ValueError(invalid_language)

    op_dag = _build_op_dag(*output_list)
    dag = op_dag.proto_dag
    inputs = op_dag.inputs

    output_buffers = []
    profiling_times = {}
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
            lib_path = ctypes.c_char_p(lib_path.encode('ascii'))
            f_name = ctypes.c_char_p((name+'_generic_cpp').encode('ascii'))
            err = test_c_op(lib_path, f_name,
                            cur_input_params, ctypes.c_size_t(num_inputs),
                            cur_output_params, ctypes.c_size_t(num_outputs),
                            eval_times_ms,
                            ctypes.c_size_t(profiling_iterations))
        elif target_language == 'cuda':
            lib_path = _make_generic_cuda(op_cuda_generic, name)
            lib_path = ctypes.c_char_p(lib_path.encode('ascii'))
            f_name = ctypes.c_char_p((name+'_generic_cuda').encode('ascii'))
            err = test_cuda_op(lib_path, f_name,
                               cur_input_params, ctypes.c_size_t(num_inputs),
                               cur_output_params, ctypes.c_size_t(num_outputs),
                               ctypes.c_uint16(_default_cuda_threads_per_block),
                               eval_times_ms,
                               ctypes.c_size_t(profiling_iterations))
        else:
            raise ValueError(invalid_language)

        profiling_times[name] = eval_times_ms
        # TODO: deallocate output buffers that are no longer needed

    outputs = []
    for out_ref in dag.dag_outputs:
        outputs.append(output_buffers[out_ref.op_index][out_ref.op_output_index])

    return outputs, profiling_times


def as_tensorflow(tensor_list):
    """
    Create a DAG of TensorFlow operators based on a DAG of OVL operators and register it with the current
    TensorFlow Graph. The inputs to the DAG must be numpy arrays or TensorFlow tensors.

    :param tensor_list: operator outputs to convert to TensorFlow tensors

    :return: A TensorFlow operator.
    """
    op_dag = _build_op_dag(*tensor_list)

    return _dag_to_tf(op_dag.proto_dag, op_dag.inputs, op_dag.grad_dags)


def _dag_to_tf(dag, inputs, grad_dags):
    output_tensors = []
    # compile all ops in the dag
    for op_index, op in enumerate(dag.operators):
        name = _op_hash(op)

        # generate code
        op_c_src, op_cuda_src, op_cuda_launch_template, op_c_generic, op_cuda_generic = \
            ExpressionDAG.generate(op, name)

        logger.debug('Compiling generic C++ for Op ' + name)
        cpu_op_lib = _make_generic_c(op_c_generic, name)
        if cuda_enabled:
            logger.debug('Compiling generic CUDA for Op ' + name)
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

        if grad_dags[op_index] is None:
            serialized_grad_dag = ''
        else:
            serialized_grad_dag = grad_dags[op_index].SerializeToString()

        tf_op = _DynamicLibOp.module().dynamic_lib(inputs=cur_inputs,
                                                   out_shapes=out_shapes,
                                                   out_types=out_tf_types,
                                                   cpu_lib_path=cpu_op_lib,
                                                   cpu_func_name=name + '_generic_cpp',
                                                   gpu_lib_path=cuda_op_lib,
                                                   gpu_func_name=name + '_generic_cuda',
                                                   serialized_grad_dag=serialized_grad_dag,
                                                   cuda_threads_per_block=_default_cuda_threads_per_block)
        output_tensors.append(tf_op)

    outputs = []
    for out_ref in dag.dag_outputs:
        outputs.append(output_tensors[out_ref.op_index][out_ref.op_output_index])

    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs
