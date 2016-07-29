# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

from ..operator import operator, gradient, OperatorOutput
from ..expression import output, output_like, position_in, \
    uint8, uint16, uint32, uint64
from ..expression import tanh as expr_tanh, exp as expr_exp, logical_and as expr_logical_and, variable, if_


def _cwise_unary(arg, func):
    out = output(arg.shape, arg.dtype)
    pos = position_in(arg.shape)
    out[pos] = func(arg[pos])

    return out


def _cwise_unary_grad(arg, grad_above, func):
    out = output(arg.shape, arg.dtype)
    pos = position_in(arg.shape)
    out[pos] = func(arg[pos], grad_above[pos])

    return out


@operator()
def neg(arg):
    if arg.dtype in [uint8, uint16, uint32, uint64]:
        raise TypeError('Cannot negate an unsigned tensor.')

    return _cwise_unary(arg, lambda x: -x)


@gradient(neg)
@operator()
def neg_grad(arg, grad_above):
    return _cwise_unary_grad(arg, grad_above, lambda x, dz: - dz)

OperatorOutput.register_magic_method('neg', neg)


@operator()
def tanh(arg):
    return _cwise_unary(arg, lambda x: expr_tanh(x))


@gradient(tanh)
@operator()
def tanh_grad(arg, grad_above):
    return _cwise_unary_grad(arg, grad_above, lambda x, dz: (1-expr_tanh(x)*expr_tanh(x))*dz)


@operator()
def sigmoid(arg):
    return _cwise_unary(arg, lambda x: 1/(1 + expr_exp(-x)))


@gradient(sigmoid)
@operator()
def sigmoid_grad(arg, grad_above):
    grad = output_like(arg)
    pos = position_in(arg.shape)

    valid_grad = expr_logical_and(arg[pos] > -50, arg[pos] < 50)
    result = variable(0, arg.dtype)
    with if_(valid_grad):
        e = expr_exp(-arg[pos])
        result <<= e/((1+e)*(1+e))

    grad[pos] = result*grad_above[pos]

    return grad


@operator()
def split(arg, split_dim=None, num_split=None):
    split_size, remainder = divmod(arg.shape[split_dim], num_split)
    if remainder != 0:
        raise ValueError('num_split must evenly divide the axis')

    out_shape = []
    for dim in range(arg.rank):
        if dim == split_dim:
            out_shape.append(split_size)
        else:
            out_shape.append(arg.shape[dim])
    pos = position_in(out_shape)

    outputs = []
    for n in range(num_split):
        outputs.append(output(out_shape, arg.dtype))
        input_pos = []
        for dim in range(arg.rank):
            if dim == split_dim:
                input_pos.append(n*split_size + pos[dim])
            else:
                input_pos.append(pos[dim])
        outputs[n][pos] = arg[input_pos]

    return outputs


@operator()
def concat(*args, **constants):
    concat_dim = constants['concat_dim']

    concat_bounds = [0]
    ref_rank = args[0].rank
    ref_shape = args[0].shape
    ref_type = args[0].dtype
    for arg in args:
        assert arg.rank == ref_rank
        assert arg.dtype == ref_type

        concat_bounds.append(concat_bounds[-1] + arg.shape[concat_dim])
        for dim in range(ref_rank):
            if dim != concat_dim:
                assert arg.shape[dim] == ref_shape[dim]

    out_shape = []
    for dim in range(ref_rank):
        if dim == concat_dim:
            out_shape.append(concat_bounds[-1])
        else:
            out_shape.append(ref_shape[dim])

    out = output(out_shape, ref_type)
    pos = position_in(out_shape)

    for arg_n, arg in enumerate(args):
        l_bound = concat_bounds[arg_n]
        u_bound = concat_bounds[arg_n + 1]
        concat_pos = pos[concat_dim]
        with if_(expr_logical_and(concat_pos >= l_bound, concat_pos < u_bound)):
            in_pos = []
            for dim in range(ref_rank):
                if dim == concat_dim:
                    in_pos.append(concat_pos - l_bound)
                else:
                    in_pos.append(pos[dim])
            out[pos] = arg[in_pos]

    return out


@gradient(split)
def split_grad(arg, *grads_above, **constants):
    split_dim = constants['split_dim']
    c = concat(*grads_above, concat_dim=split_dim)
    return c


@gradient(concat)
def concat_grad(*args, **constants):
    concat_dim = constants['concat_dim']
    inputs = args[:-1]
    num_inputs = len(inputs)
    grad_above = args[-1]
    s = split(grad_above, split_dim=concat_dim, num_split=num_inputs)

    grads = []
    for n in range(num_inputs):
        grads.append(s[n])
    return grads


def _broadcast_cwise_binary(lhs, rhs, fcn):
    if lhs.dtype != rhs.dtype:
        raise TypeError('Binary operator input dtypes must be the same.')

    if lhs.size == 1 and rhs.size > 1:
        # broadcast lhs into rhs
        out = output_like(rhs)
        lhs_pos = [0]*lhs.rank
        rhs_pos = out_pos = position_in(rhs.shape)
    elif lhs.size > 1 and rhs.size == 1:
        # broadcast rhs into lhs
        out = output_like(lhs)
        lhs_pos = out_pos = position_in(lhs.shape)
        rhs_pos = [0]*rhs.rank
    elif lhs.shape == rhs.shape:
        # component-wise without broadcasting
        out = output_like(lhs)
        lhs_pos = rhs_pos = out_pos = position_in(lhs.shape)
    else:
        raise TypeError('Binary operands must either be the same shape, or at least one must be a scalar.')

    out[out_pos] = fcn(lhs[lhs_pos], rhs[rhs_pos])

    return out


def _broadcast_cwise_binary_grad(lhs, rhs, grad, fcn):
    if lhs.dtype != rhs.dtype or lhs.dtype != grad.dtype:
        raise TypeError('Binary operator input and gradient dtypes must be the same.')

    lhs_grad = output_like(lhs)
    rhs_grad = output_like(rhs)

    if lhs.size == 1 and rhs.size > 1:
        # broadcast lhs into rhs
        lhs_pos = [0]*lhs.rank
        rhs_pos = grad_pos = position_in(rhs.shape)
    elif lhs.size > 1 and rhs.size == 1:
        # broadcast rhs into lhs
        lhs_pos = grad_pos = position_in(lhs.shape)
        rhs_pos = [0]*rhs.rank
    elif lhs.shape == rhs.shape:
        # component-wise without broadcasting
        lhs_pos = rhs_pos = grad_pos = position_in(lhs.shape)
    else:
        raise TypeError('Binary operands must either be the same shape, or at least one must be a scalar.')

    lhs_val, rhs_val = fcn(lhs[lhs_pos], rhs[rhs_pos], grad[grad_pos])
    lhs_grad[lhs_pos] = lhs_val
    rhs_grad[rhs_pos] = rhs_val

    return lhs_grad, rhs_grad


@operator()
def add(lhs, rhs):
    return _broadcast_cwise_binary(lhs, rhs, lambda x, y: x + y)


@gradient(add)
@operator()
def add_grad(lhs, rhs, grad_above):
    return _broadcast_cwise_binary_grad(lhs, rhs, grad_above, lambda x, y, dz: (dz, dz))

OperatorOutput.register_magic_method('add', add)


@operator()
def sub(lhs, rhs):
    return _broadcast_cwise_binary(lhs, rhs, lambda x, y: x - y)


@gradient(sub)
@operator()
def sub_grad(lhs, rhs, grad_above):
    return _broadcast_cwise_binary_grad(lhs, rhs, grad_above, lambda x, y, dz: (dz, -dz))

OperatorOutput.register_magic_method('sub', sub)


@operator()
def mul(lhs, rhs):
    return _broadcast_cwise_binary(lhs, rhs, lambda x, y: x * y)


@gradient(mul)
@operator()
def mul_grad(lhs, rhs, grad_above):
    return _broadcast_cwise_binary_grad(lhs, rhs, grad_above, lambda x, y, dz: (y*dz, x*dz))

OperatorOutput.register_magic_method('mul', mul)


@operator()
def div(lhs, rhs):
    return _broadcast_cwise_binary(lhs, rhs, lambda x, y: x / y)


@gradient(div)
@operator()
def div_grad(lhs, rhs, grad_above):
    return _broadcast_cwise_binary_grad(lhs, rhs, grad_above, lambda x, y, dz: (dz/y, -x/(y*y)*dz))

OperatorOutput.register_magic_method('div', div)


@operator()
def mod(lhs, rhs):
    return _broadcast_cwise_binary(lhs, rhs, lambda x, y: x % y)

OperatorOutput.register_magic_method('mod', mod)


@operator()
def equal(lhs, rhs):
    return _broadcast_cwise_binary(lhs, rhs, lambda x, y: x == y)

OperatorOutput.register_magic_method('eq', equal)


@operator()
def not_equal(lhs, rhs):
    return _broadcast_cwise_binary(lhs, rhs, lambda x, y: x != y)

OperatorOutput.register_magic_method('ne', not_equal)


@operator()
def less(lhs, rhs):
    return _broadcast_cwise_binary(lhs, rhs, lambda x, y: x < y)

OperatorOutput.register_magic_method('lt', less)


@operator()
def less_equal(lhs, rhs):
    return _broadcast_cwise_binary(lhs, rhs, lambda x, y: x <= y)

OperatorOutput.register_magic_method('le', less_equal)


@operator()
def greater(lhs, rhs):
    return _broadcast_cwise_binary(lhs, rhs, lambda x, y: x > y)

OperatorOutput.register_magic_method('gt', greater)


@operator()
def greater_equal(lhs, rhs):
    return _broadcast_cwise_binary(lhs, rhs, lambda x, y: x >= y)

OperatorOutput.register_magic_method('ge', greater_equal)
