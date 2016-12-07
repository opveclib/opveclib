# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import opveclib as ovl


def sig(arg):
    return 1/(1 + ovl.exp(-arg))


def sig_grad(arg):
    valid_grad = ovl.logical_and(arg > -50, arg < 50)
    result = ovl.variable(0, arg.dtype)
    with ovl.if_(valid_grad):
        e = ovl.exp(-arg)
        result <<= e/((1+e)*(1+e))
    return result


def tanh_grad(arg):
    t = ovl.tanh(arg)
    return 1 - t*t


@ovl.operator()
def lstm(concat, c, forget_bias=None):
    batches = concat.shape[0]
    vec_len = concat.shape[1]/4

    assert c.shape[0] == concat.shape[0]
    assert c.shape[1] == vec_len
    assert c.dtype == concat.dtype

    pos = ovl.position_in([batches, vec_len])
    cur_batch = pos[0]
    cur_elem = pos[1]

    i = concat[cur_batch, cur_elem]
    j = concat[cur_batch, cur_elem + vec_len]
    f = concat[cur_batch, cur_elem + 2*vec_len]
    o = concat[cur_batch, cur_elem + 3*vec_len]
    c_cur = c[cur_batch, cur_elem]

    new_c = ovl.output_like(c)
    new_h = ovl.output_like(c)

    new_c_cur = c_cur*sig(f + forget_bias) + sig(i) * ovl.tanh(j)

    new_c[pos] = new_c_cur
    new_h[pos] = ovl.tanh(new_c_cur) * sig(o)

    return new_c, new_h


@ovl.gradient(lstm)
@ovl.operator()
def lstm_grad(concat, c, d_new_c, d_new_h, forget_bias=None):
    batches = concat.shape[0]
    vec_len = concat.shape[1]/4

    assert c.shape[0] == concat.shape[0]
    assert c.shape[1] == vec_len
    assert c.dtype == concat.dtype

    assert d_new_c.tensor_type == c.tensor_type
    assert d_new_h.tensor_type == c.tensor_type

    pos = ovl.position_in([batches, vec_len])
    cur_batch = pos[0]
    cur_elem = pos[1]

    i = concat[cur_batch, cur_elem]
    j = concat[cur_batch, cur_elem + vec_len]
    f = concat[cur_batch, cur_elem + 2*vec_len]
    o = concat[cur_batch, cur_elem + 3*vec_len]
    c_cur = c[cur_batch, cur_elem]
    new_c_cur = c_cur*sig(f + forget_bias) + sig(i) * ovl.tanh(j)

    d_new_c_cur = d_new_c[cur_batch, cur_elem]
    d_new_h_cur = d_new_h[cur_batch, cur_elem]

    d_concat = ovl.output_like(concat)
    d_c = ovl.output_like(c)

    back_ch = d_new_c_cur + tanh_grad(new_c_cur)*sig(o)*d_new_h_cur
    d_i = ovl.tanh(j)*sig_grad(i)*back_ch
    d_j = sig(i)*tanh_grad(j)*back_ch
    d_f = c_cur*sig_grad(f+forget_bias)*back_ch
    d_c_cur = sig(f+forget_bias)*back_ch
    d_o = ovl.tanh(new_c_cur)*sig_grad(o)*d_new_h_cur

    d_concat[cur_batch, cur_elem] = d_i
    d_concat[cur_batch, cur_elem+vec_len] = d_j
    d_concat[cur_batch, cur_elem+2*vec_len] = d_f
    d_concat[cur_batch, cur_elem+3*vec_len] = d_o
    d_c[pos] = d_c_cur

    return d_concat, d_c
