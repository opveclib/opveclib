
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
import subprocess
import string
import re
from collections import namedtuple
import opcode
import inspect
from dis import findlinestarts
import six
import numpy as np
from numpy.ctypeslib import ndpointer
from .expression import TensorType, ExpressionDAG, input, OutputTensor
from .local import version, cache_directory, cuda_enabled, cuda_directory, logger, cxx
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
                        subprocess.check_output([cxx, '-fPIC', '-Wall', '-shared',
                                                 '-std=c++11', '-O2', '-Wextra', '-DGOOGLE_CUDA=1',
                                                 '-o', dynamiclibop_path,
                                                 this_directory + '/dynamiclibop.cc',
                                                 '-isystem', cuda_directory + '/include',
                                                 '-isystem', tf_include],
                                                stderr=subprocess.STDOUT,
                                                universal_newlines=True)
                    else:
                        logger.debug('*** building dynamiclibop for CPU')
                        subprocess.check_output([cxx, '-fPIC', '-Wall', '-shared',
                                                 '-std=c++11', '-O2', '-Wextra',
                                                 '-o', dynamiclibop_path,
                                                 this_directory + '/dynamiclibop.cc',
                                                 '-isystem', tf_include],
                                                stderr=subprocess.STDOUT,
                                                universal_newlines=True)
                except subprocess.CalledProcessError as exception:
                    logger.debug('c++ compiler error: ' + exception.output)
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
                grad_dag_arg_index=op.get_attr('grad_dag_arg_index')

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

                selected_inputs = []
                for arg_index in grad_dag_arg_index:
                    selected_inputs.append(grad_inputs[arg_index])

                # make sure that the input types and expected types are consistent
                for grad_input_index, grad_input in enumerate(selected_inputs):
                    received_type = TensorType.like(grad_input).as_proto()
                    expected_type = grad_dag.dag_input_types[grad_input_index]

                    if received_type != expected_type:
                        raise TypeError('Received a tensor of type: ' + str(received_type) +
                                        ', but expected a type: ' + str(expected_type) +
                                        ' at gradient input index: ' + str(grad_input_index))

                return _dag_to_tf(grad_dag, selected_inputs, grad_of_grad_dags, None)

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

    def __init__(self, dag, output_types, inputs, grad_dag, grad_dag_arg_index, from_gradient, name):
        self.inputs = inputs
        self.expression_dag = dag
        self.output_types = output_types
        self.name = name
        self.grad_dag = grad_dag
        self.grad_dag_arg_index = grad_dag_arg_index
        self.from_gradient = from_gradient

        # logger.debug('Operator created: ' + str(dag.name))

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

    def __doc__(self):
        return self.op_function.__doc__

    def __str__(self):
        return self.op_function.__name__

    def __call__(self, *inputs, **defined_constants):

        # Determine if this function is part of a gradient graph. If so, do not generate a gradient DAG.
        from_gradient = False
        for inp in inputs:
            try:
                if inp.from_gradient:
                    from_gradient = True
            except AttributeError:
                pass

            if isinstance(inp, _GradientPlaceholder):
                from_gradient = True

        func_args, func_varargs, func_keywords, func_defaults = inspect.getargspec(self.op_function)
        if func_defaults is None:
            input_names = func_args
            constants = defined_constants
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

        if len(input_names) != num_inputs and func_varargs is None:
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

        if self.grad_function is None or from_gradient:
            proto_grad_dag = None
            grad_dag_arg_index = None
        else:
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

            grad_dag = _build_op_dag(*grad_outputs)
            proto_grad_dag = grad_dag.proto_dag
            grad_dag_arg_index = []
            for inp in grad_dag.inputs:
                grad_dag_arg_index.append(grad_inputs.index(inp))

        return _Operator(expression_dag, output_types, inputs, proto_grad_dag, grad_dag_arg_index,
                         from_gradient, self.name)

    def add_grad(self, grad_function):
        if self.grad_function is None:
            self.grad_function = grad_function
        else:
            raise ValueError('Gradient function is already defined for operator ' + str(self.name) + '.')


def operator(forbid_none_valued_constants=True, name=None):
    def wrapper(op_function):
        # Use disassembler to check for reassignments to variables inside the op function

        # define compatibility function for python 2 and 3 bytecodes
        def bytecode_to_int(x):
            if six.PY2:
                return ord(x)
            elif six.PY3:
                return x

        co = op_function.__code__
        code = co.co_code
        linestarts = dict(findlinestarts(co))

        extended_arg = 0
        extended_argj = 0
        linestart_opcode_n = 0
        linestart = 0
        op_last = 0
        variable_names = set()

        # step through op codes
        cur_opcode_n = 0
        while cur_opcode_n < len(code):
            is_assignment = False
            is_variable = False
            has_lshift = False
            cur_opcode = bytecode_to_int(code[cur_opcode_n])
            if cur_opcode_n in linestarts:
                linestart_opcode_n = cur_opcode_n
                linestart = linestarts[cur_opcode_n]

            if cur_opcode == opcode.opmap['STORE_FAST']:
                is_assignment = True
                prev_opcode_n = linestart_opcode_n
                while prev_opcode_n < cur_opcode_n and not is_variable:
                    prev_opcode = bytecode_to_int(code[prev_opcode_n])
                    prev_opcode_n += 1
                    if prev_opcode >= opcode.HAVE_ARGUMENT:
                        opargj = bytecode_to_int(code[prev_opcode_n]) + \
                                 bytecode_to_int(code[prev_opcode_n+1])*256 + extended_argj
                        extended_argj = 0
                        prev_opcode_n += 2
                        if prev_opcode == opcode.EXTENDED_ARG:
                            extended_argj = opargj*65536
                        elif prev_opcode == opcode.opmap['LOAD_ATTR'] or prev_opcode == opcode.opmap['LOAD_GLOBAL']:
                            hasname = prev_opcode in opcode.hasname
                            named_var = co.co_names[opargj] == 'variable'
                            is_variable = hasname and named_var

                # check to see if prior op code is lshift
                has_lshift = op_last == opcode.opmap['INPLACE_LSHIFT']

            cur_opcode_n += 1
            if cur_opcode >= opcode.HAVE_ARGUMENT:
                oparg = bytecode_to_int(code[cur_opcode_n]) + bytecode_to_int(code[cur_opcode_n+1])*256 + extended_arg
                extended_arg = 0
                cur_opcode_n += 2

                if cur_opcode == opcode.EXTENDED_ARG:
                    extended_arg = oparg*65536
                elif cur_opcode in opcode.haslocal:
                    variable_name = co.co_varnames[oparg]
                    func_name = co.co_name

                    if is_assignment:
                        if variable_name in variable_names and not has_lshift:

                            s = '  File "' + co.co_filename + '", line ' + str(linestart)

                            raise SyntaxError('Cannot reassign to symbol "' + variable_name + '" in operator "' +
                                              func_name + '" because it refers to an OVL variable. Use the <<= operator'
                                                          ' to assign to OVL variables. \n' + s)
                        elif is_variable:
                            variable_names.add(variable_name)

            op_last = cur_opcode

        op = _OpGenerator(op_function, forbid_none_valued_constants, name)
        op.__name__ = op_function.__name__
        op.__doc__ = op_function.__doc__
        return op

    return wrapper


def gradient(op_function):
    if not isinstance(op_function, _OpGenerator):
        raise TypeError('gradient decorator argument must be a function decorated as an operator')

    def wrapper(grad_function):
        if isinstance(grad_function, _OpGenerator):
            def resolved(*args, **defaults):
                op = grad_function(*args, **defaults)

                op_outputs = []
                for output_index in range(len(op.output_types)):
                    op_outputs.append(op[output_index])

                return op_outputs
        else:
            resolved = grad_function

        op_function.add_grad(resolved)
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


def _get_expr_indices(expr_dag, expr_code):
    """
    Get indices of all read/write expression in the expression dag which is represented as a list.
    :param expr_dag: The expression dag to search for read/write expressions.
    :param expr_code: The expression code.
    :return: A list of indices with all expressions that match the expression code.
    """
    expr_indices = []
    for iExp, expr in enumerate(expr_dag.expressions):
        if expr.code == expr_code:
            expr_indices.append(iExp)
    return expr_indices


def _get_output_indices(expr_dag):
    """
    Find indices of expressions that are outputs in the expression dag.
    :param expr_dag: expression dag.
    :return: a list of output indices.
    """
    return _get_expr_indices(expr_dag, lang.OUTPUT)


def _get_input_indices(expr_dag):
    """
    Find indices of expression that are inputs in the expression dag.
    :param expr_dag: the expression dag.
    :return: a list of input indices.
    """
    return _get_expr_indices(expr_dag, lang.INPUT)


def _get_position_index(expr_dag):
    """
    Returns the position index within this expr_dag. Checks that there is one and only one POSITION expression.
    :param expr_dag: The expression dag.
    :return: The index of the position within the expression dag.
    """
    pos_indices = _get_expr_indices(expr_dag, lang.POSITION)
    assert len(pos_indices) == 1  # There is only one position definition
    return pos_indices[0]


def _get_output_io_index(expr_dag, out_expr_index):
    """
    Assumes that OUTPUT expressions appear in return order.
    :param expr_dag: The expression dag.
    :param out_expr_index: The index of the output in the expression dag.
    :return: The output argument index. Returns -1 if iOut is not an output and there are no outputs before.
    """
    assert expr_dag.expressions[out_expr_index].code == lang.OUTPUT
    io_index = -1
    for i in range(out_expr_index, 0, -1):
        if expr_dag.expressions[i].code == lang.OUTPUT:
            io_index += 1
    return io_index


def _get_output_shape(expr_dag, out_arg_index):
    """
    Returns the shape of the 'out_arg_index'-th output in the expression dag.
    :param expr_dag: expression dag
    :param out_arg_index: output index. Must be >=0 and < number of outputs.
    :return: shape of the output.
    """
    outs = _get_output_indices(expr_dag)
    assert out_arg_index < len(outs)
    return expr_dag.expressions[outs[out_arg_index]].tensor_type.shape


def _get_tensor_read_indices(expr_dag):
    """
    Get indices for the READ_TENSOR expression code.
    :param expr_dag: The expression dag.
    :return: A list of indices with a READ_TENSOR expression.
    """
    return _get_expr_indices(expr_dag, lang.READ_TENSOR)


def _get_tensor_assign_indices(expr_dag):
    """
    Get indices for the ASSIGN_TENSOR expression code.
    :param expr_dag: The expression dag.
    :return: A list of indices with ASSIGN_TENSOR expressions.
    """
    return _get_expr_indices(expr_dag, lang.ASSIGN_TENSOR)


def _get_indices_connected_to_expr_index(exp_dag, expr_indices, expr_index):
    """
    Filter 'expr_indices' if they can be traced back to 'expr_index'. Typical use cases are:
    Does a READ_TENSOR expression have connection to an input or output tensor?
    Does an ASSIGN_TENSOR expression have connection to an output tensor?
    :param exp_dag: The expression dag.
    :param expr_indices: Expression indices to be tested if they have a connection to expr_index.
    :param expr_index: An expr_index in the dag.
    :return: A sub-list of 'expr_indices' that contains only those indices of expressions that are connected to the
    expression at the index 'expr_index'. Returned indices may not be in the same order than indices in 'expr_indices'.
    """
    connected_indices = set()
    indices = []
    for index in expr_indices:
        indices.append(index)
        while len(indices) > 0:
            i = indices.pop(0)
            for op_index in exp_dag.references[i].operand_indices:
                if op_index == expr_index:
                    connected_indices.add(index)
                    # We can stop here, because we traced the op_index back to expr_index.
                    indices[:] = []
                    break
                else:
                    indices.append(op_index)
    return list(connected_indices)


def _get_indices_connected_to_expr_indices(exp_dag, from_expr_indices, to_expr_indices):
    """
    Filters the from_expr_indices down toward indices that are connected to any index within to_expr_indices.
    :param exp_dag: The expression dag.
    :param from_expr_indices: Test these indices for a connection with any of the to_expr_indices.
    :param to_expr_indices: A set of expression indices.
    :return: A set with all indices of from_expr_indices that are connected. This is a subset of from_expr_indices or
    equal to that set.
    """
    connected_indices = set()
    indices = []
    for index in from_expr_indices:
        indices.append(index)
        while len(indices) > 0:
            i = indices.pop(0)
            for op_index in exp_dag.references[i].operand_indices:
                if op_index in to_expr_indices:
                    connected_indices.add(index)
                    # We can stop here, because we traced the op_index back to expr_index.
                    indices[:] = []
                    break
                else:
                    indices.append(op_index)

    return list(connected_indices)


def _get_tensor_read_indices_for_expr_index(expr_dag, expr_index):
    """
    Returns a list of all TENSOR_READs for the expr_index, which usually is the index of an input/output tensor. An
    input tensor can only be read but not assigned to in ovl.
    :param expr_dag: The expression dag.
    :param expr_index: Usually the index in the expression dag of an input/output tensor.
    :return: A list of all expression indices that are TENSOR_READs and are connected to expr_index.
    """
    read_indices = _get_tensor_read_indices(expr_dag)
    return _get_indices_connected_to_expr_index(expr_dag, read_indices, expr_index)


def _get_indices_of_sub_expr_wout_position(expr_dag, expr_index):
    """
    Returns a list of all indices traversing the expression dag from expr_index to either an expression without inputs
    or a POSITION expression. The index for the POSITION expression is not included in the returned set.
    :param expr_dag: The expression dag.
    :param expr_index: The start index for the traversal toward inputs in the expression dag.
    :return: A set with all indices of the traversal.
    """
    sub_expr_indices = set()
    indices = [expr_index]
    while len(indices) > 0:
        i = indices.pop(0)
        # Stop at position.
        if expr_dag.expressions[i].code == lang.POSITION:
            continue
        if i not in sub_expr_indices:
            sub_expr_indices.add(i)
            indices.extend(expr_dag.references[i].operand_indices)

    return sub_expr_indices


def _get_indices_of_sub_expr(expr_dag, expr_index):
    """
    Get all indices in expression dag starting from 'expr_index' following input references.
    :param expr_dag: The expression dag.
    :param expr_index: Start the traversal toward inputs from here.
    :return: A set with all indices of the traversal starting from expr_index.
    """
    sub_expr_indices = set()
    indices = [expr_index]
    while len(indices) > 0:
        i = indices.pop(0)
        if i not in sub_expr_indices:
            sub_expr_indices.add(i)
            indices.extend(expr_dag.references[i].operand_indices)

    return sub_expr_indices


def _get_tensor_assign_indices_for_expr_index(expr_dag, expr_index):
    """
    Returns a list of all TENSOR_ASSIGNs for the expression dag, which usually an assign to an OUTPUT tensor.
    One output can have multiple writes but never a read. That is not allowed in ovl.
    :param expr_dag: The expression dag.
    :param expr_index: Usually the index of an OUTPUT tensor.
    :return: A list of all expression indices that are TENSOR_ASSIGN's and are connected to 'expr_index'.
    """
    write_indices = _get_tensor_assign_indices(expr_dag)
    return _get_indices_connected_to_expr_index(expr_dag, write_indices, expr_index)


def _match_index_in_expr_dags(expr_dag0, expr_index0, expr_dag1, expr_index1):
    """
    Compare the indexing by matching the sub-expression trees in 'expr_dag0' and 'expr_dag1'. This resolves complex
    indexing including mod, shift, etc.
    :param expr_dag0: The first expression dag.
    :param expr_index0: The index that refers to READ_TENSOR or ASSIGN_TENSOR in the first expression dag.
    :param expr_dag1: The second expression dag.
    :param expr_index1: The index that refers to READ_TENSOR or ASSIGN_TENSOR in the second expression dag.
    :return: True if the codes of all expressions match, otherwise False.
    """
    op_indices0 = expr_dag0.references[expr_index0].operand_indices
    op_indices1 = expr_dag1.references[expr_index1].operand_indices

    # Must have position 1 in operand_indices.
    assert len(op_indices0) > 1
    assert len(op_indices1) > 1

    index0 = op_indices0[1]
    index1 = op_indices1[1]

    # Initialize the indices with the index expression in both sub-trees.
    indices = [(index0, index1)]

    while len(indices) > 0:
        multi_index = indices.pop(0)
        index0 = multi_index[0]
        index1 = multi_index[1]
        expr0 = expr_dag0.expressions[index0]
        expr1 = expr_dag1.expressions[index1]

        # Check for the same expression code.
        if expr0.code != expr1.code:
            return False

        # For constants check that the have the same value.
        if expr0.code == lang.CONST_SCALAR:

            dtype0 = expr0.dtype
            dtype1 = expr1.dtype

            if dtype0 != dtype1:
                return False

            if dtype0 == lang.UINT64:
                if expr0.uint64_data != expr1.uint64_data:
                    return False
            elif dtype0 == lang.INT64:
                if expr0.sint64_data != expr1.sint64_data:
                    return False
            elif dtype0 == lang.FLOAT64:
                if expr0.double_data != expr1.double_data:
                    return False
            elif dtype0 == lang.FLOAT32:
                if expr0.float_data != expr1.float_data:
                    return False
            elif dtype0 == lang.UINT32 or dtype0 == lang.UINT16 or dtype0 == lang.UINT8:
                if expr0.uint32_data != expr1.uint32_data:
                    return False
            elif dtype0 == lang.INT32 or dtype0 == lang.INT16 or dtype0 == lang.INT8:
                if expr0.sint32_data != expr1.sint32_data:
                    return False
            else:
                raise TypeError('Cannot match unknown data type'+dtype0)

        # Continue with the expressions of this expression for each dag toward inputs.
        op_indices0 = expr_dag0.references[index0].operand_indices
        op_indices1 = expr_dag1.references[index1].operand_indices

        # The number of referenced inputs must match.
        if len(op_indices0) != len(op_indices1):
            return False

        # Add all inputs to the queue of to be processed expressions.
        for i in range(len(op_indices0)):
            indices.append((op_indices0[i], op_indices1[i]))

    return True


def _eliminate_duplicates(l):
    """
    Removes duplicates from a list while maintaining the order of the elements (stable).
    :param l: The input list.
    :return: List with removed duplicates.
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


class _MergeRef(namedtuple('_MergeRef', ['to_op_index', 'to_in_expr_index', 'to_in_arg_index', 'from_op_index',
                                         'from_out_expr_index', 'from_out_arg_index'])):
    """
    Merge reference refer to indices in the operation dag. The 'to_op_index' and 'from_op_index' encode a dependency
    between an output tensor of the from-op that is used as input tensor in the to-op. The 'to_in_expr_index' refers to
    the INPUT expression definition within the expression dag of the to-op. The 'to_in_arg_index' contains the input
    argument index as defined in the signature of the to-op. The 'from_out_expr_index' refers to the OUTPUT expression
    definition within the expression dag of the from-op. The 'from_out_arg_index' is the output argument index of the
    tensor in the from-op.
    :param to_op_index: The index of the to-op in the operator dag.
    :param to_in_expr_index: The index of the INPUT expression defining the tensor in the expression dag of the to-op.
    :param to_in_arg_index: The index of the input argument of this tensor in the to-op.
    :param from_op_index: The index of the from-op in the operator dag.
    :param from_out_expr_index: The index of the OUTPUT expression defining the tensor in the expression dag of the
    from-op.
    :param from_out_arg_index: The index of the output argument of the from-op.
    :return: A merge reference.
    """
    __slots__ = () # empty slots

    def same(self, ref):
        """
        Two merge reference are the same if all their indices match. This method does NOT test for object equality.
        :param ref: The other merge reference.
        :return: True if they are the same, otherwise false.
        """
        return self.to_op_index == ref.to_op_index \
               and self.to_in_expr_index == ref.to_in_expr_index \
               and self.to_in_arg_index == ref.to_in_arg_index \
               and self.from_op_index == ref.from_op_index \
               and self.from_out_expr_index == ref.from_out_expr_index \
               and self.from_out_arg_index == ref.from_out_arg_index


def _get_merge_refs_for_op_dag(proto_op_dag):
    """
    Creates a list of operators with their arguments that can be merged into single operators. Notice that merge
    information is given per argument.
    :param proto_op_dag: the protobuf representation of the operator dag.
    :return: merge information that contains a list of merge_refs for operators and their input/output arguments and it
    contains a list of merge_names of operator names (ambiguous) together with argument indices as strings which is ONLY
    useful for debugging.
    """
    ops = proto_op_dag.operators     # get operators, each operator contains an expression dag
    outs = proto_op_dag.dag_outputs
    refs = proto_op_dag.references    # references of the operator dag

    # Walk the dag from each output to inputs and compute information about merging of ops.
    inputs = []
    staged = set()
    merge_refs = []    # Holds the indices of ops in tuples.

    # For each output tensor of the operator dag.
    for output in outs:

        output_index = output.op_index  # operator index of this output
        inputs.append(output_index)     # add output index
        staged.clear()                  # remove all staged operator indices.
        staged.add(output_index)        # mark this operator index as staged.

        while len(inputs) > 0:
            to_op_index = inputs.pop(0)
            to_exp_dag = ops[to_op_index]
            to_ref = refs[to_op_index]
            to_workgroup_shape = to_exp_dag.workgroup_shape

            for to_in_arg_index, input in enumerate(to_ref.input_refs):
                from_op_index = input.op_index

                # Do not consider leaf nodes.
                if input.is_leaf:
                    continue

                # Append if not staged.
                if from_op_index not in staged:
                    staged.add(from_op_index)
                    inputs.append(from_op_index)

                expr = to_exp_dag.expressions[input.dag_input_index]
                from_exp_dag = ops[from_op_index]
                from_workgroup_shape = from_exp_dag.workgroup_shape
                input_shape_of_out = expr.tensor_type.shape
                output_indices = _get_output_indices(from_exp_dag)
                output_index = output_indices[input.op_output_index]
                output_shape_of_in = _get_output_shape(from_exp_dag, input.op_output_index)

                match = to_workgroup_shape == from_workgroup_shape
                if not match:
                    logger.debug('Non-matching workgroup shapes for %s input [%d] and %s output [%d].'
                                 % (to_exp_dag.name, to_in_arg_index, from_exp_dag.name, input.op_output_index))
                    continue

                #assert input_shape_of_out == output_shape_of_in # Must always match in well defined dags.
                # Because of broadcasting that we currently do not handle.
                match = input_shape_of_out == output_shape_of_in
                if not match:
                    logger.debug('Non-matching tensor shapes (boradcasting) for %s input [%d] and %s output [%d].'
                                 % (to_exp_dag.name, to_in_arg_index, from_exp_dag.name, input.op_output_index))
                    continue

                # get the indexing pattern for the output to this input.
                tensor_write_indices = _get_tensor_assign_indices_for_expr_index(from_exp_dag, output_index)
                match = len(tensor_write_indices) == 1

                if not match:
                    logger.debug('Multiple writes to %s output [%d]' % (from_exp_dag.name, input.op_output_index))
                    continue

                # if there are multiple different write patterns we cannot merge.
                tensor_write_index = tensor_write_indices[0]
                tensor_read_indices = _get_tensor_read_indices_for_expr_index(to_exp_dag, input.dag_input_index)
                for read_index in tensor_read_indices:
                    match = _match_index_in_expr_dags(from_exp_dag, tensor_write_index, to_exp_dag, read_index)
                    if not match:
                        break

                if not match:
                    logger.debug('No-matching read/write index pattern for %s input [%d] and %s output [%d]'
                                 % (to_exp_dag.name, to_in_arg_index, from_exp_dag.name, input.op_output_index))
                    continue

                merge_refs.append(_MergeRef(to_op_index=to_op_index,
                                            to_in_expr_index=input.dag_input_index,
                                            to_in_arg_index=to_in_arg_index,
                                            from_op_index=from_op_index,
                                            from_out_expr_index=output_index,
                                            from_out_arg_index=input.op_output_index))

    # Eliminate duplicates in merge_refs and merge_names.
    return _eliminate_duplicates(merge_refs)


def _group_merge_refs(proto_op_dag, merge_refs):
    """
    Takes the protbuf representation of the operator dag together with the merge references and converts them into
    grouped merge references, which is pairs of (to-op-index, from-op-index).
    Two ops can be merged if:
      - All inputs of the to-op come from one and the same from-op.
      - OR inputs can come from external inputs.
    :param proto_op_dag: the protobuf representation of the operator dag.
    :param merge_refs: merge references.
    :return: pairs of (to-op-index, from-op-index).
    """

    # Create a set with triplets (to_op_index, from_op_index, from_out_arg_index) as elements.
    merge_group_info = set()
    for merge_ref in merge_refs:
        merge_group_info.add((merge_ref.to_op_index, merge_ref.from_op_index, merge_ref.from_out_arg_index))

    group_merge_refs = []

    for i, reference in enumerate(proto_op_dag.references):
        if len(reference.input_refs) == 0:
            continue

        from_op_index = reference.input_refs[0].op_index

        # Input is in refs and has an reference as leaf node with op_index = 0.
        if from_op_index == i:
            continue

        to_op_index = i
        do_group = True

        for in_ref in reference.input_refs:
            # If ref is an external input, then continue.
            if in_ref.is_leaf:
                continue

            # The input comes from another op, we cannot merge in this case.
            if in_ref.op_index != from_op_index:
                do_group = False
                break

            # This input tensor cannot be merged because it does not appear in the merge info.
            from_out_arg_index = in_ref.op_output_index
            if (to_op_index, from_op_index, from_out_arg_index) not in merge_group_info:
                do_group = False
                break

        # If we can group append the tuple to the list.
        if do_group:
            group_merge_refs.append((to_op_index, from_op_index))

    return group_merge_refs


_IndicesInfo = namedtuple('_IndicesInfo', ['used_only_in_to', 'used_only_in_to_io',
                                           'used_in_general', 'used_in_general_io',
                                           'external_inputs_for_to', 'external_inputs_for_to_io',
                                           'outputs_removed_in_from',
                                           'new_out_indices', 'input_to_ouput_index'])


def _get_indices_info(proto_op_dag, to_op_index, from_op_index):
    """
    Returns index information about:
    - ASSIGN_TENSOR indices within the from-operator that are used only in the to-operator.
    - ASSIGN_TENSOR indices within the from-operator that are used in the to-operator and other operators.
    - Their associated indices as they appear in the listing of input/output arguments for the to-operator and
    from-operator.
    - The INPUT indices for any external inputs of the to-operator.
    - The updated output indices according to the VARIABLE definitions within the from-operator as they are used within
    the merged operator for the to-operator reads.
    - An input to output index mapping.
    :param proto_op_dag: The protobuf description of an operator dag.
    :param to_op_index: The to-operator index within the operator dag.
    :param from_op_index: The from-operator index within the operator dag.
    :return: Index information.
    """
    references = proto_op_dag.references
    ops = proto_op_dag.operators
    outs = proto_op_dag.dag_outputs

    used_indices_io = set()
    external_inputs_for_to_io = set()
    external_outputs = dict() # Key is the operator index and the value is the set of output indices.

    for out in outs:
        if out.op_index not in external_outputs:
            external_outputs[out.op_index] = set()
        external_outputs[out.op_index].add(out.op_output_index)

    for i, ref in enumerate(references[to_op_index].input_refs):
        if ref.op_index == from_op_index:
            used_indices_io.add(ref.op_output_index)
        if ref.is_leaf:
            external_inputs_for_to_io.add(i)

    # Output tensors of the from-op could be used by ops other than the to-op.
    used_in_general_io = set()
    for i, reference in enumerate(references):
        if i == to_op_index or i == from_op_index:
            continue
        for ref in reference.input_refs:
            if ref.op_index == from_op_index \
                    and ref.op_output_index in used_indices_io:
                used_in_general_io.add(ref.op_output_index)

    # Or output tensors of the from-op can be used as external output.
    if from_op_index in external_outputs:
        used_in_general_io |= external_outputs[from_op_index]

    used_only_in_to_io = used_indices_io - used_in_general_io

    out_indices = _get_output_indices(ops[from_op_index])
    used_only_in_to = set()
    used_in_general = set()

    # Mapping of from-op output indices to their new output index.
    new_out_indices = list()
    new_out_index = 0
    for i in range(0, len(out_indices)):
        if i in used_only_in_to_io:
            used_only_in_to.add(out_indices[i])
            new_out_indices.append(0)
        elif i in used_in_general_io:
            used_in_general.add(out_indices[i])
            new_out_indices.append(new_out_index)
            new_out_index += 1
        else:
            new_out_indices.append(new_out_index)
            new_out_index += 1

    # Mapping from input to output indices.
    input_to_output_index = dict()
    for i, ref in enumerate(references[to_op_index].input_refs):
        if ref.op_index == from_op_index:
            input_to_output_index[i] = ref.op_output_index

    external_inputs_for_to = set()
    input_indices = _get_input_indices(ops[to_op_index])
    for i, input_index in enumerate(input_indices):
        if i in external_inputs_for_to_io:
            external_inputs_for_to.add(input_index)

    outputs_removed_in_from = set()
    outputs_removed_in_from |= used_only_in_to_io
    if from_op_index in external_outputs:
        # Exclude all those indices that are used as external output.
        outputs_removed_in_from -= external_outputs[from_op_index]

    return _IndicesInfo(used_only_in_to=used_only_in_to,
                        used_only_in_to_io=used_only_in_to_io,
                        used_in_general=used_in_general,
                        used_in_general_io=used_in_general_io,
                        external_inputs_for_to=external_inputs_for_to,
                        external_inputs_for_to_io=external_inputs_for_to_io,
                        outputs_removed_in_from=outputs_removed_in_from,
                        new_out_indices=new_out_indices,
                        input_to_ouput_index=input_to_output_index)


def _add_variable_const_expr(merged_expr_dag, old_expr):
    """
    Adds a VARIABLE expression together with a CONST_SCALAR expression.
    :param merged_expr_dag: The merged expression dag.
    :param old_expr: The old expression used to read-out the data type.
    :return: None.
    """
    old_dtype = old_expr.tensor_type.dtype

    const_scalar = lang.Expression()
    const_scalar.code = lang.CONST_SCALAR
    const_scalar.dtype = old_dtype

    var = lang.Expression()
    var.code = lang.VARIABLE
    var.dtype = old_dtype

    # Set the initial value.
    if old_dtype == lang.FLOAT64:
        const_scalar.double_data.append(0)
    elif old_dtype == lang.FLOAT32 or old_dtype == lang.FLOAT16:
        const_scalar.float_data.append(0)
    elif old_dtype == lang.UINT64:
        const_scalar.uint64_data.append(0)
    elif old_dtype == lang.INT64:
        const_scalar.sint64_data.append(0)
    elif old_dtype == lang.UINT32 or old_dtype == lang.UINT16 or old_dtype == lang.UINT8:
        const_scalar.uint32_data.append(0)
    elif old_dtype == lang.INT32 or old_dtype == lang.INT16 or old_dtype == lang.INT8:
        const_scalar.sint32_data.append(0)
    else:
        raise TypeError('Tried to add variable for unknown type '+old_dtype)

    head_expr = merged_expr_dag.expressions.add()
    head_expr.CopyFrom(const_scalar)
    merged_expr_dag.references.add()

    head_expr = merged_expr_dag.expressions.add()
    head_expr.CopyFrom(var)

    # The VARIABLE expression refers to the CONST_SCALAR expression.
    merged_expr_dag.references.add().operand_indices.extend([len(merged_expr_dag.references) - 2])


def _merge_expr_dags(to_expr_dag, from_expr_dag, indices_info):
    """
    Merges two expression dags. Two expression dags are merged using the following:
    A) For the 'from_exp_dag' we do:
        - For tensors not used in the to-op leave them as is.
        - For tensors ONLY used in the to-op we replace the OUTPUT expression by VARIABLE expression and the
        ASSIGN_TENSOR expression by an ASSIGN_VARIABLE expression.
        - For tensors used in the to-op AND in another op we add a VARIABLE expression and an ASSIGN_VARIABLE expression.
    B) For the 'to_expr_dag' we do:
        - We replace READ_TENSOR expressions that refer to reads from tensors in the to-op through a READ_VARIABLE
        expression. Notice that there are several other READ_TENSOR expressions, e.g. reading the POSITION value or
        reading from external inputs that can not be replaced.
    :param to_expr_dag: The expression dag of the to-operator.
    :param from_expr_dag: The expression dag of the from-operator.
    :param indices_info: The index information used for the merge.
    :return: The merged expression dag.
    """
    used_only_in_to = indices_info.used_only_in_to
    used_in_general = indices_info.used_in_general
    input_to_output_index = indices_info.input_to_ouput_index
    external_inputs_for_to = indices_info.external_inputs_for_to

    merged_expr_dag = lang.ExpressionDAG()
    assign_indices = _get_tensor_assign_indices(from_expr_dag)
    del_indices = set()
    only_assign_indices = _get_indices_connected_to_expr_indices(from_expr_dag, assign_indices, used_only_in_to)
    general_assign_indices = _get_indices_connected_to_expr_indices(from_expr_dag, assign_indices, used_in_general)

    for only_index in only_assign_indices:
        # Purge the index of the ASSIGN_TENSOR expression but not the ASSIGN_TENSOR expression itself.
        index_index = from_expr_dag.references[only_index].operand_indices[1]
        del_indices |= _get_indices_of_sub_expr_wout_position(from_expr_dag, index_index)

    # Offset per index of expressions of from-op's expression dag.
    offsets = np.zeros(len(from_expr_dag.expressions), dtype=int)
    offset = 0
    output_to_expr_index = dict()
    # Counters for the number of inputs and outputs of the merged op.
    input_count = 0
    output_count = 0

    # ******************************************************************************************************************
    # For the 'from' expression dag do:
    # - For tensors not used in the 'to' op leave everything as is.
    # - For tensors ONLY used in the 'to' op replace the OUTPUT by VARIABLE and ASSIGN_TENSOR by ASSIGN_VARIABLE.
    # - For tensors used in the 'to' op AND in another op add VARIABLE and ASSIGN_VARIABLE.
    # ******************************************************************************************************************
    for i, expr in enumerate(from_expr_dag.expressions):

        # Do not copy expressions contained in the del_indices set.
        if i in del_indices:
            offset -= 1
            offsets[i] = offset
            continue

        # Replace the OUTPUT expression by a VARIABLE expression.
        if i in used_only_in_to:
            _add_variable_const_expr(merged_expr_dag, expr)
            offset += 1
            io = _get_output_io_index(from_expr_dag, i)
            output_to_expr_index[io] = len(merged_expr_dag.expressions) - 1
            offsets[i] = offset
            continue

        # Replace the ASSIGN_TENSOR expression by an ASSIGN_VARIABLE expression.
        if i in only_assign_indices:
            head_expr = merged_expr_dag.expressions.add()
            var_assign = lang.Expression()
            var_assign.code = lang.ASSIGN_VARIABLE
            head_expr.CopyFrom(var_assign)
            i0 = from_expr_dag.references[i].operand_indices[0]
            i1 = from_expr_dag.references[i].operand_indices[2]
            operand_indices = [int(offsets[i0]) + i0, int(offsets[i1]) + i1]
            merged_expr_dag.references.add().operand_indices.extend(operand_indices)
            offsets[i] = offset
            continue

        # Copy the original expression.
        head_expr = merged_expr_dag.expressions.add()
        head_expr.CopyFrom(from_expr_dag.expressions[i])

        # Set the io_index in the sequence of the INPUT definitions. This assumes that input definitions appear
        # according to the order in the signature.
        if head_expr.code == lang.INPUT:
            head_expr.io_index = input_count
            input_count += 1

        # Set the io_index in the sequence of the OUTPUT definitions. This assumes that output definitions appear
        # in the order of the return statement.
        if head_expr.code == lang.OUTPUT:
            head_expr.io_index = output_count
            output_count += 1

        # Set updated references for the original expression.
        operand_indices = []
        for iRef in from_expr_dag.references[i].operand_indices:
            operand_indices.append(int(offsets[iRef]) + iRef)
        merged_expr_dag.references.add().operand_indices.extend(operand_indices)

        # Add a VARIABLE expression to the OUTPUT expression.
        if i in used_in_general:
            # Set the offset HERE as opposed to the end to refer to the OUTPUT expression and not the VARIABLE
            # expression!
            offsets[i] = offset
            _add_variable_const_expr(merged_expr_dag, expr)
            offset += 2
            io = _get_output_io_index(from_expr_dag, i)
            output_to_expr_index[io] = len(merged_expr_dag.expressions) - 1
            continue

        # Add an ASSIGN_VARIABLE.
        if i in general_assign_indices:
            # Set the offset HERE as opposed to the end to refer to the ASSIGN_TENSOR expression and not the
            # ASSIGN_VARIABLE expression!
            offsets[i] = offset
            head_expr = merged_expr_dag.expressions.add()
            var_assign = lang.Expression()
            var_assign.code = lang.ASSIGN_VARIABLE
            head_expr.CopyFrom(var_assign)
            i0 = from_expr_dag.references[i].operand_indices[0]
            i1 = from_expr_dag.references[i].operand_indices[2]
            # Find output argument index for output at expression index i0.
            io = _get_output_io_index(from_expr_dag, i0)
            operand_indices = [output_to_expr_index[io], int(offsets[i1]) + i1]
            merged_expr_dag.references.add().operand_indices.extend(operand_indices)
            offset += 1
            continue

        # In all other cases set the offset here.
        offsets[i] = offset

    # ******************************************************************************************************************
    # For the 'to' expression do:
    # - Replace READ_TENSOR through the corresponding VARIABLE_READ
    # Read tensors that are excluded:
    # - Reads from external input variables.
    # - Reads that appear within ASSIGN_TENSOR statements.
    # ******************************************************************************************************************
    to_pos_index = _get_position_index(to_expr_dag)
    input_indices = _get_input_indices(to_expr_dag)
    expr_to_input_index = dict()
    for input_index in input_indices:
        expr_to_input_index[input_index] = to_expr_dag.expressions[input_index].io_index

    input_indices = set(input_indices)
    input_indices -= external_inputs_for_to
    del_indices = set()
    del_indices.add(to_pos_index)
    del_indices |= input_indices
    # Get all TENSOR_READ indices.
    read_indices = set(_get_tensor_read_indices(to_expr_dag))
    for read_index in read_indices:
        tensor_input_index = to_expr_dag.references[read_index].operand_indices[0]
        if tensor_input_index in input_indices:
            index_index = to_expr_dag.references[read_index].operand_indices[1]
            del_indices |= _get_indices_of_sub_expr_wout_position(to_expr_dag, index_index)
            del_indices.add(read_index)

    # Do not remove indices that are used in TENSOR_ASSIGN statements.
    assign_indices = _get_tensor_assign_indices(to_expr_dag)
    for assign_index in assign_indices:
        index_index = to_expr_dag.references[assign_index].operand_indices[1]
        assign_index_tree = _get_indices_of_sub_expr_wout_position(to_expr_dag, index_index)
        del_indices -= assign_index_tree
        read_indices -= assign_index_tree

    # Do not replace READ_TENSOR from external inputs.
    delete_from_read = set()
    for read_index in read_indices:
        tensor_input_index = to_expr_dag.references[read_index].operand_indices[0]
        if tensor_input_index in external_inputs_for_to:
            delete_from_read |= _get_indices_of_sub_expr_wout_position(to_expr_dag, read_index)
    read_indices -= delete_from_read

    merge_pos_index = _get_position_index(merged_expr_dag)
    offset = len(merged_expr_dag.expressions)
    offsets = np.empty(len(to_expr_dag.expressions), dtype=int)
    offsets.fill(offset)
    outputs_for_to  = _get_output_indices(to_expr_dag)

    for i, expr in enumerate(to_expr_dag.expressions):
        if i in del_indices:
            offset -= 1
        else:
            head_expr = merged_expr_dag.expressions.add()
            head_expr.CopyFrom(to_expr_dag.expressions[i])

            if i in external_inputs_for_to:
                head_expr.io_index = input_count
                input_count += 1

            if i in outputs_for_to:
                head_expr.io_index = output_count
                output_count += 1

            operand_indices = []
            for iRef in to_expr_dag.references[i].operand_indices:
                if iRef == to_pos_index:
                    operand_indices.append(merge_pos_index)
                elif iRef in read_indices:
                    input_index = to_expr_dag.references[iRef].operand_indices[0]  # reference to tensor
                    operand_indices.append(output_to_expr_index[input_to_output_index[expr_to_input_index[input_index]]])
                else:
                    operand_indices.append(int(offsets[iRef]) + iRef)

            merged_expr_dag.references.add().operand_indices.extend(operand_indices)

        offsets[i] = offset

    # Set the name and workgroup shape of the merged expression dag.
    merged_expr_dag.workgroup_shape.extend(from_expr_dag.workgroup_shape)
    merged_expr_dag.name = from_expr_dag.name + '_' + to_expr_dag.name

    return merged_expr_dag


def _merge_op_dag(proto_op_dag):
    """
    Merges the operator dag by decreasing the depth of the operator dag and increasing the depth of the expression dag.
    :param op_dag: The protobuf format of the operator dag.
    :return: A merged operator dag in protobuf format.
    """

    # Initialize the merged op-dag and the merge refs and their grouping.
    merged_op_dag = proto_op_dag
    group_merge_refs = _group_merge_refs(merged_op_dag, _get_merge_refs_for_op_dag(merged_op_dag))

    # While we can merge ops in the dag.
    while len(group_merge_refs) > 0:
        ops = merged_op_dag.operators
        refs = merged_op_dag.references
        merge_ref = group_merge_refs.pop(0)
        to_op_index = merge_ref[0]
        from_op_index = merge_ref[1]

        logger.debug('Merging ' + ops[from_op_index].name + ' and ' + ops[to_op_index].name)

        indices_info = _get_indices_info(merged_op_dag, to_op_index, from_op_index)
        merged_expr_dag = _merge_expr_dags(ops[to_op_index], ops[from_op_index], indices_info)

        new_out_indices = indices_info.new_out_indices
        external_inputs_for_to_io = indices_info.external_inputs_for_to_io
        outputs_removed_in_from = indices_info.outputs_removed_in_from
        output_from_wout_to_num = len(_get_output_indices(ops[from_op_index])) - len(indices_info.used_only_in_to)

        # Build up the newly merged operator dag.
        new_merged_op_dag = lang.OperatorDAG()

        # Update the operators.
        for i, op in enumerate(ops):
            # Replace the from operator by the merged operator.
            if i == from_op_index:
                new_merged_op_dag.operators.add().CopyFrom(merged_expr_dag)
            # Copy any op that is neither the from-operator nor the to-operator.
            elif i != to_op_index:
                new_merged_op_dag.operators.add().CopyFrom(op)

        # Update the input references to operators.
        for i, ref in enumerate(refs):
            # Exclude references for the to-operator.
            if i == to_op_index:
                continue

            # Update all other references.
            ref_list = lang.OperatorDAG.OperatorInputReferences()

            for in_ref in ref.input_refs:

                proto_ref = lang.OperatorDAG.OperatorInputReference()
                proto_ref.is_leaf = in_ref.is_leaf
                op_index = in_ref.op_index
                # Input index is the same input from the op dag.
                proto_ref.dag_input_index = in_ref.dag_input_index

                if op_index == from_op_index:
                    proto_ref.op_output_index = new_out_indices[in_ref.op_output_index]
                elif op_index == to_op_index:
                    proto_ref.op_output_index = output_from_wout_to_num + in_ref.op_output_index
                else:
                    proto_ref.op_output_index = in_ref.op_output_index

                if op_index > to_op_index:
                    op_index -= 1
                elif op_index == to_op_index:
                    op_index = from_op_index

                proto_ref.op_index = op_index
                ref_list.input_refs.add().CopyFrom(proto_ref)

            if i == from_op_index:
                # Uses the to_op_index in refs to ge the input_refs!
                for input_index, in_ref in enumerate(refs[to_op_index].input_refs):
                    # Add the input reference only if this input of the 'to-op' has not yet been mapped to an output of
                    # the 'from-op'.
                    if input_index in external_inputs_for_to_io:
                        proto_ref = lang.OperatorDAG.OperatorInputReference()
                        proto_ref.is_leaf = in_ref.is_leaf
                        op_index = in_ref.op_index
                        if op_index > to_op_index:
                            op_index -= 1
                        elif op_index == to_op_index:
                            op_index = from_op_index
                        proto_ref.op_index = op_index
                        proto_ref.op_output_index = in_ref.op_output_index
                        proto_ref.dag_input_index = in_ref.dag_input_index
                        ref_list.input_refs.add().CopyFrom(proto_ref)

            new_merged_op_dag.references.add().CopyFrom(ref_list)

        # Create the outputs of the merged op.
        for i, dag_output in enumerate(merged_op_dag.dag_outputs):
            proto_out = lang.OperatorDAG.DAGOutputReference()
            op_index = dag_output.op_index
            proto_out.op_output_index = dag_output.op_output_index
            if op_index == to_op_index:
                proto_out.op_output_index += output_from_wout_to_num
            if op_index == from_op_index:
                # Subtract as many from the op-output-index as has been removed that are smaller or equal to this index.
                sub_value = 0
                for out_remove_io in outputs_removed_in_from:
                    if out_remove_io <= proto_out.op_output_index:
                        sub_value += 1
                proto_out.op_output_index -= sub_value

            if op_index > to_op_index:
                op_index -= 1
            elif op_index == to_op_index:
                op_index = from_op_index
            proto_out.op_index = op_index

            new_merged_op_dag.dag_outputs.add().CopyFrom(proto_out)

        merged_op_dag = new_merged_op_dag
        group_merge_refs = _group_merge_refs(merged_op_dag, _get_merge_refs_for_op_dag(merged_op_dag))

    # Copy the input types only if the merged op dag differs from the initially provided op dag.
    if merged_op_dag is not proto_op_dag:
        for dag_input in proto_op_dag.dag_input_types:
            proto_type = TensorType.like(dag_input).as_proto()
            merged_op_dag.dag_input_types.add().CopyFrom(proto_type)

    return merged_op_dag


def _make_generic_c(src, name):
    # look for generic c++ shared library in the operator cache
    generic_cpp_so_path = os.path.join(cache_directory, name + '_generic_cpp.so')

    if not os.path.exists(generic_cpp_so_path):
        # logger.debug('Compiling generic C++ for Op ' + name)

        generic_cpp_path = os.path.join(cache_directory, name + '_generic_cpp.cpp')
        with open(generic_cpp_path, 'w') as f:
            f.write(src)

        this_file_path = os.path.abspath(__file__)
        this_directory = os.path.split(this_file_path)[0]
        try:
            subprocess.check_output([cxx, '-fPIC', '-std=c++11', '-g', '-pedantic',
                                     '-Wall', '-Wextra',
                                     '-I'+this_directory,
                                     '-shared',
                                     '-o', generic_cpp_so_path, generic_cpp_path],
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True)
        except subprocess.CalledProcessError as exception:
            logger.debug('c++ compiler error: ' + exception.output)
            raise

    return generic_cpp_so_path


def _make_generic_cuda(src, name):
    # look for generic cuda shared library in the operator cache
    generic_cuda_so_path = os.path.join(cache_directory, name + '_generic_cuda.so')
    if not os.path.exists(generic_cuda_so_path):
        # logger.debug('Compiling generic CUDA for Op ' + name)
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


def evaluate(output_list, target_language='cpp', opt_level=3):
    """
    Evaluate a collection of OVL operator, mainly used for testing. This function uses a test operator
    function for running the generated generic version of the operator so it does not depend on an external
    execution runtime. This also means that this function only works for operators whose inputs are numpy arrays.

    :param output_list: The outputs to evaluate
    :param target_language: 'cpp' or 'cuda'
    :param opt_level: Optimization level.

    :return:  A list of numpy arrays for each operator output in output_list
    """
    evaluated_outputs = profile(output_list, target_language=target_language,
                                profiling_iterations=1, opt_level=opt_level)[0]

    if len(evaluated_outputs) == 1:
        return evaluated_outputs[0]
    else:
        return evaluated_outputs


def profile(output_list, target_language, profiling_iterations, opt_level):
    """
    Evaluate a collection of OVL operator, mainly used for testing. This function uses a test operator
    function for running the generated generic version of the operator so it does not depend on an external
    execution runtime. This also means that this function only works for operators whose inputs are numpy arrays.

    :param output_list: The outputs to evaluate
    :param target_language: 'cpp' or 'cuda'
    :param profiling_iterations: Number of times to run this operator for profiling purposes.
     Must be a positive int.
    :param opt_level: optimization level

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
            import tensorflow as tf
            this_file_path = os.path.abspath(__file__)
            this_directory = os.path.split(this_file_path)[0]
            tf_include = tf.sysconfig.get_include()

            # build the test framework library
            cc_path = os.path.join(this_directory, 'testcop.cc')

            try:
                subprocess.check_output([cxx, '-fPIC', '-Wall', '-shared',
                                         '-std=c++11', '-Ofast', '-Wextra',
                                         '-I'+this_directory,
                                         '-I'+cache_directory,
                                         '-isystem', tf_include,
                                         '-o', testlib_path, cc_path],
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
            except subprocess.CalledProcessError as exception:
                logger.debug('c++ compiler error: ' + exception.output)
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
                # to compiling. The default compiler, g++, has no concept of this, so
                # we have to do an extra device code link step with a dummy link file
                linko_path = os.path.join(cache_directory, 'link.o')
                subprocess.check_output([nvcc_path, '-dlink', '-Xcompiler', '-fPIC',
                                         '-o', linko_path, o_path],
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
                subprocess.check_output([cxx, '-shared',
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
    if opt_level >= 3:
        dag = _merge_op_dag(dag)
    inputs = op_dag.inputs

    output_buffers = []
    profiling_times = {}
    # compile all ops in the dag
    for op_index, op in enumerate(dag.operators):
        name = _op_hash(op)

        # generate code
        op_c_src, op_cuda_src, op_c_generic, op_cuda_generic = \
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


def as_tensorflow(tensor_list, opt_level=3):
    """
    Create a DAG of TensorFlow operators based on a DAG of OVL operators and register it with the current
    TensorFlow Graph. The inputs to the DAG must be numpy arrays or TensorFlow tensors.

    :param tensor_list: operator outputs to convert to TensorFlow tensors

    :return: A TensorFlow operator.
    """
    op_dag = _build_op_dag(*tensor_list)
    grad_dag_arg_index_list = []
    for op in op_dag.operators:
        grad_dag_arg_index_list.append(op.grad_dag_arg_index)

    return _dag_to_tf(op_dag.proto_dag, op_dag.inputs, op_dag.grad_dags, grad_dag_arg_index_list)


def _dag_to_tf(dag, inputs, grad_dags, grad_dag_arg_index_list):
    output_tensors = []
    # compile all ops in the dag
    for op_index, op in enumerate(dag.operators):
        name = _op_hash(op)

        # generate code
        op_c_src, op_cuda_src, op_c_generic, op_cuda_generic = \
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
            grad_dag_arg_index = []
        else:
            serialized_grad_dag = grad_dags[op_index].SerializeToString()
            grad_dag_arg_index = grad_dag_arg_index_list[op_index]

        tf_op = _DynamicLibOp.module().dynamic_lib(inputs=cur_inputs,
                                                   out_shapes=out_shapes,
                                                   out_types=out_tf_types,
                                                   cpu_lib_path=cpu_op_lib,
                                                   cpu_func_name=name + '_generic_cpp',
                                                   gpu_lib_path=cuda_op_lib,
                                                   gpu_func_name=name + '_generic_cuda',
                                                   serialized_grad_dag=serialized_grad_dag,
                                                   grad_dag_arg_index=grad_dag_arg_index,
                                                   cuda_threads_per_block=_default_cuda_threads_per_block)
        output_tensors.append(tf_op)

    outputs = []
    for out_ref in dag.dag_outputs:
        outputs.append(output_tensors[out_ref.op_index][out_ref.op_output_index])

    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs
