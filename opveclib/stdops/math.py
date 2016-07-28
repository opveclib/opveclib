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


@operator()
def neg(arg):
    if arg.dtype in [uint8, uint16, uint32, uint64]:
        raise TypeError('Cannot negate an unsigned tensor.')

    out = output(arg.shape, arg.dtype)
    pos = position_in(arg.shape)
    out[pos] = -arg[pos]
    return out


@gradient(neg)
@operator()
def div_grad(arg, grad_above):
    grad = output(arg.shape, arg.dtype)
    pos = position_in(arg.shape)
    grad[pos] = -grad_above[pos]
    return grad


OperatorOutput.register_magic_method('neg', neg)
