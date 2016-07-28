
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


class OperatorOutput(object):
    """
    Class which represents an un-evaluated output tensor, used for building lazily evaluated DAGs of operators
    """

    # raise operator priority so that numpy does not try to use its own operator
    __array_priority__ = 1
    _builtin_ops = {}

    @staticmethod
    def register_magic_method(key, value):
        OperatorOutput._builtin_ops[key] = value

    def __init__(self, parent, index):
        if not isinstance(parent, _Operator):
            raise TypeError('parent must be an Operator')
        if not isinstance(index, int):
            raise TypeError('index must be an int')

        self.parent = parent
        self.index = index
        self.tensor_type = parent.output_types[index]
        self.shape = parent.output_types[index].shape
        self.dtype = parent.output_types[index].dtype
        self.size = self.tensor_type.size

    def __add__(self, other):
        return OperatorOutput._builtin_ops['add'](self, other)

    def __radd__(self, other):
        return OperatorOutput._builtin_ops['add'](other, self)

    def __sub__(self, other):
        return OperatorOutput._builtin_ops['sub'](self, other)

    def __rsub__(self, other):
        return OperatorOutput._builtin_ops['sub'](other, self)

    def __mul__(self, other):
        return OperatorOutput._builtin_ops['mul'](self, other)

    def __rmul__(self, other):
        return OperatorOutput._builtin_ops['mul'](other, self)

    # python 2
    def __div__(self, other):
        return OperatorOutput._builtin_ops['div'](self, other)

    def __rdiv__(self, other):
        return OperatorOutput._builtin_ops['div'](other, self)

    # python 3
    def __truediv__(self, other):
        return OperatorOutput._builtin_ops['div'](self, other)

    def __rtruediv__(self, other):
        return OperatorOutput._builtin_ops['div'](other, self)

    def __mod__(self, other):
        return OperatorOutput._builtin_ops['mod'](self, other)

    def __rmod__(self, other):
        return OperatorOutput._builtin_ops['mod'](other, self)

    def __eq__(self, other):
        return OperatorOutput._builtin_ops['eq'](self, other)

    def __ne__(self, other):
        return OperatorOutput._builtin_ops['ne'](self, other)

    def __lt__(self, other):
        return OperatorOutput._builtin_ops['lt'](self, other)

    def __le__(self, other):
        return OperatorOutput._builtin_ops['le'](self, other)

    def __gt__(self, other):
        return OperatorOutput._builtin_ops['gt'](self, other)

    def __ge__(self, other):
        return OperatorOutput._builtin_ops['ge'](self, other)

    def __neg__(self):
        return OperatorOutput._builtin_ops['neg'](self)

    @staticmethod
    def __bool__(self):
        raise SyntaxError('Cannot resolve operator values at interpretation time.')


class _GradientPlaceholder(object):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _Operator(object):
    """
    Class which is extended to define a new operator and its gradient.
    """

    # raise operator priority so that numpy does not try to use its own operator
    __array_priority__ = 1

    def __init__(self, dag, output_types, inputs, grad_dag, name):
        self.inputs = inputs
        self.expression_dag = dag
        self.output_types = output_types
        self.name = name
        self.grad_dag = grad_dag

        logger.debug('Operator created: ' + str(dag.name))

    def __getitem__(self, item):
        return OperatorOutput(self, item)

    def check_binary(self):
        if len(self.output_types) != 1:
            raise SyntaxError('Cannot use binary infix operators on multi-output operators. '
                              'Explicitly index an output.')

    def __add__(self, other):
        self.check_binary()
        return self[0] + other

    def __radd__(self, other):
        self.check_binary()
        return other + self[0]

    def __sub__(self, other):
        self.check_binary()
        return self[0] - other

    def __rsub__(self, other):
        self.check_binary()
        return other - self[0]

    def __mul__(self, other):
        self.check_binary()
        return self[0] * other

    def __rmul__(self, other):
        self.check_binary()
        return other * self[0]

    # python 2
    def __div__(self, other):
        self.check_binary()
        return self[0] / other

    def __rdiv__(self, other):
        self.check_binary()
        return other / self[0]

    # python 3
    def __truediv__(self, other):
        self.check_binary()
        return self[0] / other

    def __rtruediv__(self, other):
        self.check_binary()
        return other / self[0]

    def __mod__(self, other):
        self.check_binary()
        return self[0] % other

    def __rmod__(self, other):
        self.check_binary()
        return other % self[0]

    def __eq__(self, other):
        self.check_binary()
        return self[0] == other

    def __ne__(self, other):
        self.check_binary()
        return self[0] != other

    def __lt__(self, other):
        self.check_binary()
        return self[0] < other

    def __le__(self, other):
        self.check_binary()
        return self[0] <= other

    def __gt__(self, other):
        self.check_binary()
        return self[0] > other

    def __ge__(self, other):
        self.check_binary()
        return self[0] >= other

    def __neg__(self):
        self.check_binary()
        return -self[0]


def _resolve_output(x):
    """
    Resolve whether or not an object is an OperatorOutput. Converts single-output Operators to an OperatorOutput

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
    elif not isinstance(x, OperatorOutput):
        raise TypeError('Only operator outputs can be used to build an op dag. Received a ' + str(type(x)))
    else:
        return x


def _op_hash(op):
    return 'f' + hashlib.sha224(op.SerializeToString() + version.encode('utf-8')).hexdigest()


class _OpGenerator(object):
    def __init__(self, op_function, forbid_none_valued_constants, name):
        self.op_function = op_function
        self.forbid_none_valued_constants = forbid_none_valued_constants

        # name this op based on it's function name unless a name is supplied to the operator decorator
        if name is None:
            self.name = self.op_function.__name__
        else:
            self.name = self.name

        self.grad_function = None
        self.num_grad_args = None

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

        if self.forbid_none_valued_constants:
            for key in constants.keys():
                if constants[key] is None:
                    raise ValueError(self.name + ' argument ' + key + ' is None, which implies an unset constant.\n'
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
                                str(inp_n + 1) + ' in the call to ' + self.name + '.  '
                                'Should this argument be passed as a constant (keyword argument) instead?')

        num_inputs = len(input_types)

        if len(input_names) != num_inputs:
            err_msg = '\n'
            err_msg += self.name + ' function signature expects ' + str(len(input_names)) + \
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
        expression_dag.name = self.name

        if self.grad_function is None:
            grad_dag = None
        else:
            if len(output_types) + len(input_names) != self.num_grad_args:
                raise SyntaxError('Gradient function must have number of inputs equal to the sum of the number of '
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
                if isinstance(grad_output, _Operator) and len(grad_output.output_types) != 1:
                    raise TypeError('A multi-output operator was returned from a gradient function, but the meaning '
                                    'of this is ambiguous: explicitly index each output from operator: ' +
                                    str(grad_output.name))
                cur_input_type = input_types[grad_output_index]
                cur_grad_output_type = TensorType.like(_resolve_output(grad_output))
                if cur_input_type != cur_grad_output_type:
                    raise TypeError('Gradient output index ' + str(grad_output_index) + ', with TensorType: ' +
                                    str(cur_grad_output_type) + ' is inconsistent with operator input index ' +
                                    str(grad_output_index) + ', with TensorType: ' + str(cur_input_type))

            grad_dag = _build_op_dag(*grad_outputs).proto_dag

        return _Operator(expression_dag, output_types, inputs, grad_dag, self.name)

    def add_grad(self, grad_function, num_grad_args):
        if self.grad_function is None:
            self.grad_function = grad_function
            self.num_grad_args = num_grad_args
        else:
            raise ValueError('Gradient function is already defined for operator ' + str(self.name) + '.')


def operator(forbid_none_valued_constants=True, name=None):
    def wrapper(op_function):
        if inspect.getargspec(op_function).keywords is not None:
            raise SyntaxError('Operator functions cannot accept keyword arguments without default values.')

        if inspect.getargspec(op_function).varargs is not None:
            raise NotImplementedError('Operator functions cannot accept varags.')

        return _OpGenerator(op_function, forbid_none_valued_constants, name)
    return wrapper


def gradient(op_function):
    if not isinstance(op_function, _OpGenerator):
        raise TypeError('gradient decorator argument must be a function decorated as an operator')

    def wrapper(grad_function):
        func_args, func_varargs, func_keywords, func_defaults = inspect.getargspec(op_function.op_function)
        if func_defaults is None:
            func_input_names = func_args
            func_constants = {}
        else:
            num_func_defaults = len(func_defaults)
            func_input_names = func_args[:-num_func_defaults]
            func_constants = dict(zip(func_args[-num_func_defaults:], func_defaults))

        if isinstance(grad_function, _OpGenerator):
            grad_args, grad_varargs, grad_keywords, grad_defaults = inspect.getargspec(grad_function.op_function)

            def resolved(*args, **defaults):
                op = grad_function(*args, **defaults)

                op_outputs = []
                for output_index in range(len(op.output_types)):
                    op_outputs.append(op[output_index])

                return op_outputs
        else:
            grad_args, grad_varargs, grad_keywords, grad_defaults = inspect.getargspec(grad_function)
            resolved = grad_function

        if grad_defaults is None:
            grad_input_names = grad_args
            grad_constants = {}
        else:
            num_grad_defaults = len(grad_defaults)
            grad_input_names = grad_args[:-num_grad_defaults]
            grad_constants = dict(zip(grad_args[-num_grad_defaults:], grad_defaults))

        if func_constants != grad_constants:
            raise SyntaxError('Constant argument names and default values must be identical for '
                              'the op function and its gradient.')

        if func_input_names != grad_input_names[:len(func_input_names)]:
            raise SyntaxError('Gradient function must have same initial argument names as the op function.')

        op_function.add_grad(resolved, len(grad_input_names))
        return resolved

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


def _get_output_indices(expr_dag):
    """
    Find indices of expressions that are outputs in the expression dag.
    :param expr_dag: expression dag
    :return: a list of output indices
    """
    outs = []
    io_last = -1
    for iExpr, expr in enumerate(expr_dag.expressions._values):
        if expr.code == lang.OUTPUT:
            outs.append(iExpr)
            assert expr.io_index>io_last # Checks the ordering constraint!
            io_last = expr.io_index
    return outs


def _get_output_shape(expr_dag, iOut):
    """
    Returns the shape of the iOut-th output in the expression dag.
    :param expr_dag: expression dag
    :param iOut: output index. Must be >=0 and < number of outputs.
    :return: shape of the output.
    """
    outs = _get_output_indices(expr_dag)
    assert iOut < len(outs)
    return expr_dag.expressions._values[outs[iOut]].tensor_type.shape


def _get_expr_indices(expr_dag, expr_code):
    """
    Get indices of all read/write expression in the expression dag which is represented as a list.
    :param expr_dag: The expression dag to search for read/write expressions.
    :param expr_code: The expression code.
    :return: A list of indices with all expressions that match the expression code.
    """
    expr_indices = []
    exprs = expr_dag.expressions._values
    for iExp, expr in enumerate(exprs):
        if expr.code == expr_code:
            expr_indices.append(iExp)
    return expr_indices


def _get_tensor_read_indices(expr_dag):
    """
    Get indices for the READ_TENSOR expression code.
    :param expr_dag: The expression dag.
    :return: A list of indices with READ_TENSOR expressions.
    """
    return _get_expr_indices(expr_dag, lang.READ_TENSOR)


def _get_tensor_write_indices(expr_dag):
    """
    Get indices for the ASSIGN_TENSOR expression code.
    :param expr_dag: The expression dag.
    :return: A list of indices with ASSIGN_TENSOR expressions.
    """
    return _get_expr_indices(expr_dag, lang.ASSIGN_TENSOR)


def _get_indices_connected_to_expr_index(exp_dag, expr_indices, expr_index):
    """
    Get indices from expr_indices that can be traced back to expr_index.  A typical use case is to see if a READ_TENSOR
    or ASSIGN_TENSOR expression is linked to an input or output tensor of an operation.
    :param exp_dag: The expression dag.
    :param expr_indices: The expression indices to test if they have a connection to expr_index.
    :param expr_index: An expr_index in the dag. Assumes expr_index>=0 and expr_index<len(exp_dag.expressions._values).
    :return: A sub-list of expr_indices that contains only those indices of expressions that are connected to the
    expression at the index expr_index. We DO NOT guarantee for the returned indices to be in the same order than the
    indices in expr_indices.
    """
    connected_indices = set()
    indices = []
    for rw_index in expr_indices:
        indices.append(rw_index)
        while len(indices) > 0:
            i = indices.pop(0)
            for op_index in exp_dag.references._values[i].operand_indices:
                if op_index==expr_index:
                    connected_indices.add(rw_index)
                    # We can stop here, because we traced the op_index back to expr_index.
                    indices[:] = []
                    break
                else:
                    indices.append(op_index)
    return list(connected_indices)

def _get_tensor_read_indices_for_expr_index(expr_dag, expr_index):
    """
    Returns a list of all TENSOR_READs for the expr_index, which usually is an input/output tensor.
    An input tensor can only be read but not assigned to in ovl.
    :param expr_dag: The expression dag.
    :param expr_index: Usually the index of an input/output tensor.
    :return: A list of all expression indices that are TENSOR_READs and are connected to expr_index.
    """
    read_indices = _get_tensor_read_indices(expr_dag)
    return _get_indices_connected_to_expr_index(expr_dag, read_indices, expr_index)

def _get_tensor_write_indices_for_expr_index(expr_dag, expr_index):
    """
    Returns a list of all TENSOR_ASSIGNs for teh expr_index, which usually is an input/output tensor.
    One output can have multiple writes but never a read. That is not allowed in ovl.
    :param expr_dag: The expression dag.
    :param expr_index: Usually the index of an input/output tensor.
    :return: A list of all expression indices that are TENSOR_ASSIGNs and are connected to expr_index.
    """
    write_indices = _get_tensor_write_indices(expr_dag)
    return _get_indices_connected_to_expr_index(expr_dag, write_indices, expr_index)

def _refs_pos(expr_dag, expr_index):
    """
    Returns True if the input references of expr_index have a POSITION code among them otherwise False.
    :param expr_dag: The expression dag.
    :param expr_index: The expression index to inspect.
    :return: True if POSITION code is among the input references otherwise False.
    """
    for op_index in expr_dag.references._values[expr_index].operand_indices:
        expr = expr_dag.expressions._values[op_index]
        if expr.code == lang.POSITION:
            return True

    return False


def _get_worker_indices(expr_dag, tensor_rw_index):
    """
    We define the worker index of TENSOR_READ or a TENSOR_ASSIGN as follows:
    (i)  it must be traced back to a POSITION code.
    (ii) since the codes contain the flattened index we re-construct the original index position in row-major order
         based on the ADD tree when flatting the index.
    For instance, assume our tensor_rw_index and those codes before the tensor_rw_index express tensor[i0,i1,i2,...],
    and i2 is traced back to a POSITION code, then we return 2.
    Because there could be multiple READS or ASSIGNS we return a list of indices.
    :param expr_dag: The expression dag.
    :param tensor_rw_index: The index of the read/write code to look into.
    :return: A list of worker indices.
    """
    worker_indices = [] # empty if none found, no worker reads from this tensor
    # Walk the tree from the root to the leaves and then return that leave which has the POSITION code.
    indices = [(tensor_rw_index, 0)]
    while len(indices) > 0:
        multi_index = indices.pop(0)
        expr_index = multi_index[0]
        arg_index = multi_index[1]
        op_indices = expr_dag.references._values[expr_index].operand_indices
        for i, op_index in enumerate(op_indices):
            wasAddOp = expr_dag.expressions._values[expr_index].code == lang.ADD
            expr = expr_dag.expressions._values[op_index]
            if wasAddOp:
                arg_index = arg_index + i # We assume that ADD codes are binary i=0 or i=1!
            # Do not follow further READ_TENSOR instances.
            if expr.code != lang.READ_TENSOR:
                indices.append((op_index, arg_index))
            else:
                # Immediate read from POSITION otherwise we do not need to follow another tensor read.
                if _refs_pos(expr_dag, op_index):
                    worker_indices.append(arg_index)
    return worker_indices


def _match_index_in_expr_dags(expr_dag0, expr_index0, expr_dag1, expr_index1):
    """
    Compare the indexing by matching the sub-expression trees in expr_dag0 and expr_dag1. This captures also more
    complex indexing patterns as given by mod, shift etc.
    :param expr_dag0: The first expression dag.
    :param expr_index0: The index that refers to the READ_TENSOR or ASSIGN_TENSOR statement in the first dag.
    :param expr_dag1: The second expression dag.
    :param expr_index1: The index that refers to the READ_TENSOR or ASSIGN_TENSOR statement in the second dag.
    :return: True if the codes of all expressions match, otherwise False.
    """
    op_indices0 = expr_dag0.references._values[expr_index0].operand_indices
    op_indices1 = expr_dag1.references._values[expr_index1].operand_indices
    assert len(op_indices0) > 1 # Must have sub-expression tree for index at pos 1
    assert len(op_indices1) > 1 # Must have sub-expression tree for index at pos 1
    index0  = op_indices0[1]
    index1  = op_indices1[1]
    indices = [(index0, index1)] # Initialize the indices with the index expression in both sub-trees.
    while len(indices) > 0:
        multi_index = indices.pop(0)
        index0      = multi_index[0]
        index1      = multi_index[1]
        expr0       = expr_dag0.expressions._values[index0]
        expr1       = expr_dag1.expressions._values[index1]

        if expr0.code != expr1.code:
            return False

        if expr0.code == lang.CONST_SCALAR:
            if expr0.sint64_data != expr1.sint64_data:
                return False

        op_indices0 = expr_dag0.references._values[index0].operand_indices
        op_indices1 = expr_dag1.references._values[index1].operand_indices

        if len(op_indices0) != len(op_indices1):
            return False

        for i in range(len(op_indices0)):
            indices.append((op_indices0[i], op_indices1[i]))

    return True


def _eliminate_duplicates(l):
    """
    Removes duplicates from a list while maintaining the order of elements (stable).
    :param l: The input list.
    :return: List where duplicates were removed.
    """
    # TODO: Improve this implementation of making lists unique while maintaining the original order (stable).
    duplicate = [False] * len(l)
    for i1 in range(len(l)):
        e = l[i1]
        for i2 in range(i1+1,len(l)):
            if e == l[i2]:
                duplicate[i2] = True
    unique = []
    for i, d in zip(range(len(duplicate)), duplicate):
        if not d:
            unique.append(l[i])
    return unique


class _MergeRef(namedtuple('_MergeRef', ['to_op_index', 'to_in_arg_index', 'from_op_index', 'from_out_arg_index'])):
    """
    Merge reference referring to the indices in the operation dag to/from indices and the input/output argument index
    as defined in the signature of the operation. This input/output argument index has a correspondence in the
    expression dag of the operator.
    :param to_op_index: As from the programmers view this is a to operation with arguments supplied by a from operation.
    :param to_in_arg_index: The index of the input argument in the to operation.
    :param from_op_index: The operation that arguments are supplied from.
    :param from_out_arg_index: The index of the output argument of the from operation.
    :return: A to merge reference.
    """
    __slots__ = () # empty slots
    def same(self, ref):
        """
        Two merge reference are the same if all their indices match. This method does NOT measure object equality.
        :param ref: The other merge reference.
        :return: True if they are the same or False otherwise.
        """
        return self.to_op_index == ref.to_op_index \
               and self.to_in_arg_index == ref.to_in_arg_index \
               and self.from_op_index == ref.from_op_index \
               and self.from_out_arg_index == ref.from_out_arg_index

_MergeInfo = namedtuple('_MergeInfo', ['merge_refs','merge_names'])


def _get_merge_refs_for_op_dag(op_dag):
    """
    Creates a list of operators with their arguments that can be merged into single operators. Notice that merge
    information is given per argument.
    :param op_dag: operator dag.
    :return: merge information that contains a list of merge_refs for operators and their input/output arguments and it
    contains a list of merge_names of operator names (ambiguous) together with argument indices as strings which is ONLY
    useful for debugging.
    """
    dag     = op_dag.proto_dag          # get the dag
    ops     = dag.operators._values     # get operators, each operator contains an expression dag
    outs    = dag.dag_outputs._values   # get outputs of the operator dag
    refs    = dag.references._values    # references of the operator dag

    # Walk the dag from each output to inputs and compute information about merging of ops.
    inputs      = []    # List (ab)used as queue.
    staged      = set() # Set that holds ops staged for processing.
    merge_refs  = []    # Holds the indices of ops in tuples.
    merge_names = []    # For debugging hold the name of ops (those are not unique).

    for output in outs:

        iOutput = output.op_index   # operator index of this output
        inputs.append(iOutput)      # add output index
        staged.clear()              # remove all staged operator indices.
        staged.add(iOutput)         # mark this operator index as staged.

        while len(inputs) > 0:
            to_op_index         = inputs.pop(0)
            to_exp_dag          = ops[to_op_index]
            to_ref              = refs[to_op_index]
            to_workgroup_shape  = to_exp_dag.workgroup_shape

            #for to_in_arg_index, input in zip(range(len(to_ref.input_refs)), to_ref.input_refs):
            for to_in_arg_index, input in enumerate(to_ref.input_refs):
                from_op_index = input.op_index

                # Do not consider leaf nodes.
                if input.is_leaf:
                    continue

                # Append if not staged.
                if from_op_index not in staged:
                    staged.add(from_op_index)
                    inputs.append(from_op_index)

                expr                    = to_exp_dag.expressions._values[input.dag_input_index]
                from_exp_dag            = ops[from_op_index]
                from_workgroup_shape    = from_exp_dag.workgroup_shape
                input_shape_of_out      = expr.tensor_type.shape
                output_indices          = _get_output_indices(from_exp_dag)
                output_index            = output_indices[input.op_output_index]
                output_shape_of_in      = _get_output_shape(from_exp_dag, input.op_output_index)

                match = to_workgroup_shape == from_workgroup_shape
                assert input_shape_of_out == output_shape_of_in # Must always match in well defined dags.

                #print('%s in [%d], %s out [%d]' % (to_exp_dag.name, to_in_arg_index, from_exp_dag.name, input.op_output_index))

                if not match:
                    #print('Workgroup shapes dont match.')
                    continue

                # get the indexing pattern for the output to this input.
                tensor_write_indices = _get_tensor_write_indices_for_expr_index(from_exp_dag, output_index)
                match = len(tensor_write_indices) == 1

                if not match:
                    #print('Multiple writes to output tensor.')
                    continue

                # if there are multiple different write patterns we cannot merge.
                tensor_write_index = tensor_write_indices[0]
                tensor_read_indices = _get_tensor_read_indices_for_expr_index(to_exp_dag, input.dag_input_index)
                for read_index in tensor_read_indices:
                    match = _match_index_in_expr_dags(from_exp_dag, tensor_write_index, to_exp_dag, read_index)
                    if not match:
                        break

                if not match:
                    #print('Non match read/write index pattern')
                    #print('Read pattern: ', index_read_pattern)
                    #print('Write pattern: ', index_write_pattern)
                    continue

                merge_refs.append(_MergeRef(to_op_index=to_op_index,
                                         to_in_arg_index=to_in_arg_index,
                                         from_op_index=from_op_index,
                                         from_out_arg_index=input.op_output_index))
                merge_names.append((to_exp_dag.name + ' in [%d]' % to_in_arg_index,
                                 from_exp_dag.name + ' out [%d]' % input.op_output_index))

    # Duplicates are eliminated in merge_refs and merge_names
    merge_refs = _eliminate_duplicates(merge_refs)
    merge_names = _eliminate_duplicates(merge_names)
    return _MergeInfo(merge_refs=merge_refs, merge_names=merge_names)


def _merge_op_dag(op_dag):
    merge_refs = _get_merge_refs_for_op_dag(op_dag)
    # TODO: Apply the merge_refs to the op_dag by fusing expression dags.
    # TODO: When merging expression dags with loops/<<= insert a temporary for each output/input pair.
    # TODO: Need to keep gradient dags in mind.
    return op_dag


def _make_generic_c(src, name):
    # look for generic c++ shared library in the operator cache
    generic_cpp_so_path = os.path.join(cache_directory, name + '_generic_cpp.so')

    if not os.path.exists(generic_cpp_so_path):
        logger.debug('Compiling generic C++ for Op ' + name)

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
        logger.debug('Compiling generic CUDA for Op ' + name)
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

        cpu_op_lib = _make_generic_c(op_c_generic, name)
        if cuda_enabled:
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
