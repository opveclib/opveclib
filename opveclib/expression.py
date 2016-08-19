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
import string
import re
import six

import numpy as np
from . import language_pb2 as lang


class DType(object):
    """
    Data type object which indicates a low level type.
    """
    _np_lookup = {
        np.float16: lang.FLOAT16,
        np.float32: lang.FLOAT32,
        np.float64: lang.FLOAT64,
        np.int8: lang.INT8,
        np.int16: lang.INT16,
        np.int32: lang.INT32,
        np.int64: lang.INT64,
        np.uint8: lang.UINT8,
        np.uint16: lang.UINT16,
        np.uint32: lang.UINT32,
        np.uint64: lang.UINT64}

    _ctypes_lookup = {
        lang.FLOAT32: ctypes.c_float,
        lang.FLOAT64: ctypes.c_double,
        lang.INT8: ctypes.c_int8,
        lang.INT16: ctypes.c_int16,
        lang.INT32: ctypes.c_int32,
        lang.INT64: ctypes.c_int64,
        lang.UINT8: ctypes.c_uint8,
        lang.UINT16: ctypes.c_uint16,
        lang.UINT32: ctypes.c_uint32,
        lang.UINT64: ctypes.c_uint64
    }

    _cstr_lookup = {
        lang.FLOAT32: 'float',
        lang.FLOAT64: 'double',
        lang.INT8: 'int8_t',
        lang.INT16: 'int16_t',
        lang.INT32: 'int32_t',
        lang.INT64: 'int64_t',
        lang.UINT8: 'uint8_t',
        lang.UINT16: 'uint16_t',
        lang.UINT32: 'uint32_t',
        lang.UINT64: 'uint64_t'
    }

    _tensorflow_lookup = {
        lang.FLOAT32: 'float',
        lang.FLOAT64: 'double',
        lang.INT8: 'int8',
        lang.INT16: 'int16',
        lang.INT32: 'int32',
        lang.INT64: 'int64',
        lang.UINT8: 'uint8',
        lang.UINT16: 'uint16'
    }

    def __init__(self, dtype):

        if type(dtype) is DType:
            self.proto_dtype = dtype.proto_dtype
        elif dtype in list(DType._np_lookup.keys()):
            # find index by equality, rather than equivalency
            index = list(DType._np_lookup.keys()).index(dtype)
            self.proto_dtype = list(DType._np_lookup.values())[index]
        elif type(dtype) is int:
            if dtype not in list(lang.DType.values()):
                raise ValueError('dtype ' + str(dtype) + ' is not valid.')
            else:
                self.proto_dtype = dtype
        else:
            raise TypeError('dtype ' + str(dtype) + ' is not valid.')

    def __eq__(self, other):
        if type(other) is not DType:
            return False
        else:
            return self.proto_dtype == other.proto_dtype

    def __hash__(self):
        return id(self.proto_dtype)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return lang.DType.Name(self.proto_dtype)

    def as_numpy(self):
        return list(DType._np_lookup.keys())[list(DType._np_lookup.values()).index(self.proto_dtype)]

    def as_ctypes(self):
        return DType._ctypes_lookup[self.proto_dtype]

    def as_cstr(self):
        return DType._cstr_lookup[self.proto_dtype]

    def as_tensorflow(self):
        return DType._tensorflow_lookup[self.proto_dtype]

    def as_proto(self):
        return self.proto_dtype

#: The half-precision floating point DType
float16 = DType(lang.FLOAT16)
#: The single precision floating point DType
float32 = DType(lang.FLOAT32)
#: The double precision floating point DType
float64 = DType(lang.FLOAT64)
#: The 8 bit signed integer DType
int8 = DType(lang.INT8)
#: The 16 bit signed integer DType
int16 = DType(lang.INT16)
#: The 32 bit signed integer DType
int32 = DType(lang.INT32)
#: The 64 bit signed integer DType
int64 = DType(lang.INT64)
#: The 8 bit unsigned integer DType
uint8 = DType(lang.UINT8)
#: The 16 bit unsigned integer DType
uint16 = DType(lang.UINT16)
#: The 32 bit unsigned integer DType
uint32 = DType(lang.UINT32)
#: The 64 bit unsigned integer DType
uint64 = DType(lang.UINT64)

supported_types = [float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64]


class TensorType(object):
    """
    A tensor is defined by its data type and its shape.
    """
    def __init__(self, shape, dtype):
        """

        :param shape: The tensor shape
        :param dtype: The data type
        :return: A tensor type object
        """
        self.dtype = DType(dtype)

        if type(shape) is int:
            self.shape = [shape]
        else:
            self.shape = []
            for elem in shape:
                if elem is None:
                    raise TypeError('All dimensions must be defined.')
                elif type(elem) is not int:
                    raise TypeError('Shape must be an iterable of ints.')
                else:
                    self.shape.append(elem)

        for elem in self.shape:
            if elem <= 0:
                raise ValueError('All tensor dimensions must be positive, but got: ' + str(self.shape))

        self.size = 1
        for elem in self.shape:
            self.size *= elem

        self.rank = len(self.shape)

        self._proto_tensor_type = lang.TensorType()
        self._proto_tensor_type.dtype = self.dtype.proto_dtype
        self._proto_tensor_type.shape.extend(self.shape)

    def __eq__(self, other):
        if not isinstance(other, TensorType):
            return False
        else:
            return self._proto_tensor_type == other.as_proto()

    def __ne__(self, other):
        if type(other) is not TensorType:
            return True
        else:
            return self._proto_tensor_type != other.as_proto()

    def __str__(self):
        return str(self._proto_tensor_type)

    @staticmethod
    def like(other):
        """
        Resolve the TensorType the argument

        :param other: The input object
        :return: A TensorType like the input object
        """
        try:
            other_shape = other.shape
        except AttributeError:
            other_shape = other.get_shape().as_list()

        try:
            other_dtype = other.dtype.as_numpy_dtype().dtype
        except AttributeError:
            other_dtype = other.dtype

        return TensorType(other_shape, other_dtype)

    @staticmethod
    def from_proto(proto):
        """
        Recover TensorType object from protocol buffer serialization

        :param proto: the protobuf
        :return: A TensorType object
        """
        return TensorType(proto.shape, proto.dtype)

    def as_proto(self):
        """
        Serialize this object as a protocol buffer

        :return: A protobuf
        """
        tt = lang.TensorType()
        tt.CopyFrom(self._proto_tensor_type)
        return tt


def _list_to_str(x):
    out = ''
    for i, cur in enumerate(x):
        out += str(cur)
        if i < len(x) - 1:
            out += ', '
    return out


class ExpressionDAG(object):
    """
    Singleton object for keeping track of expressions in the order in which they are defined. Expressions must register themselves
     with the ExpressionDAG upon construction with the append() method.
    """
    exprs = []
    expr_ids = []
    workgroup_shape = None
    num_outputs = 0
    num_inputs = 0

    @staticmethod
    def io_types():
        input_types = ExpressionDAG.num_inputs*[None]
        output_types = ExpressionDAG.num_outputs*[None]

        found_count = 0
        for expr in ExpressionDAG.exprs:
            if expr.proto_expr.code == lang.INPUT:
                input_types[expr.proto_expr.io_index] = TensorType.from_proto(expr.proto_expr.tensor_type)
                found_count += 1
            elif expr.proto_expr.code == lang.OUTPUT:
                output_types[expr.proto_expr.io_index] = TensorType.from_proto(expr.proto_expr.tensor_type)
                found_count += 1
            if found_count == ExpressionDAG.num_inputs + ExpressionDAG.num_outputs:
                break

        return input_types, output_types


    @staticmethod
    def clear():
        """
        Clear all currently tracked expressions.
        :return:
        """
        ExpressionDAG.exprs = []
        ExpressionDAG.expr_ids = []
        ExpressionDAG.workgroup_shape = None
        ExpressionDAG.num_outputs = 0
        ExpressionDAG.num_inputs = 0

    @staticmethod
    def append(item):
        """
        Append an item to the expression list.
        :param item: The expression to append.
        :return: None
        """
        if not issubclass(item.__class__, _Expression):
            raise TypeError('Can only append expressions.')

        if type(item) is PositionTensor:
            if ExpressionDAG.workgroup_shape is None:
                ExpressionDAG.workgroup_shape = item.proto_expr.uint32_data
            else:
                raise ValueError('Already defined the position tensor.')

        if type(item) is OutputTensor:
            if item.proto_expr.io_index != ExpressionDAG.num_outputs:
                raise ValueError('Trying to add outputs to expression dag out of order')

            ExpressionDAG.num_outputs += 1

        if type(item) is InputTensor:
            if item.proto_expr.io_index != ExpressionDAG.num_inputs:
                raise ValueError('Trying to add inputs to expression dag out of order')

            ExpressionDAG.num_inputs += 1

        # Assign names to each expression as they get appended
        if item.name is None:
            if type(item) is InputTensor:
                item.name = '(*in'+str(item.proto_expr.io_index)+')'
            elif type(item) is OutputTensor:
                item.name = '(*out'+str(item.proto_expr.io_index)+')'
            elif type(item) is PositionTensor:
                item.name = 'position'
            elif type(item) is _ConstScalar:
                item.name = str(item.value())
            elif type(item) is _ConstTensor:
                item.name = '{'+_list_to_str(item.to_array().tolist())+'}'
            else:
                item.name = 'e'+str(len(ExpressionDAG.exprs))

        ExpressionDAG.exprs.append(item)
        ExpressionDAG.expr_ids.append(id(item))

    @staticmethod
    def remove_endif():
        """
        find and remove the most recent _EndIf expression, used for continuing if blocks
        :return: None
        """
        found_endif = False
        for i in range(len(ExpressionDAG.exprs)):
            if type(ExpressionDAG.exprs[-i-1]) is _EndIf:
                found_endif = True
                del(ExpressionDAG.exprs[-i-1])
                del(ExpressionDAG.expr_ids[-i-1])
                break
        if found_endif is False:
            raise SyntaxError('Could not find prior if block')

    @staticmethod
    def as_proto():
        """
        Serialize the current ExpressionDAG as a protocol buffer
        :return: the protobuf
        """
        if ExpressionDAG.workgroup_shape is None:
            raise ValueError('Workgroup shape must be defined with "position_in" function.')

        proto = lang.ExpressionDAG()
        proto.workgroup_shape.extend(ExpressionDAG.workgroup_shape)
        for i, expr in enumerate(ExpressionDAG.exprs):
            proto.expressions.add().CopyFrom(expr.proto_expr)

            operand_indices = []
            for input_expr in expr.input_exprs:
                operand_indices.append(ExpressionDAG.expr_index(input_expr))

            proto.references.add().operand_indices.extend(operand_indices)

        # Reorder op dag to make sure that all elseif conditionals are positioned
        # before entering the if block
        if_block_start = []
        needs_reordering = []

        for i, expr in enumerate(proto.expressions):

            if expr.code is lang.IF:
                if_block_start.append(i)
                needs_reordering.append([])
            elif expr.code is lang.ELSEIF:
                # recursively find all conditional dependencies that need to be reordered
                def find_reorders(x):
                    for ref in proto.references[x].operand_indices:
                        if ref > if_block_start[-1]:
                            find_reorders(ref)
                    needs_reordering[-1].append(x)
                conditional_index = proto.references[i].operand_indices[0]
                find_reorders(conditional_index)
            elif expr.code is lang.ENDIF:
                new_to_old_index = {}
                num_reorders = len(needs_reordering[-1])
                if num_reorders > 0:
                    reorder_count = 0
                    for cur_index in range(if_block_start[-1], needs_reordering[-1][-1] + 1):
                        if cur_index in needs_reordering[-1]:
                            new_to_old_index[if_block_start[-1] + reorder_count] = cur_index
                            reorder_count += 1
                        else:
                            new_to_old_index[cur_index + num_reorders - reorder_count] = cur_index

                    def new_to_old(x):
                        if x in list(new_to_old_index.keys()):
                            return new_to_old_index[x]
                        else:
                            return x

                    old_to_new_index = dict((v, k) for k, v in new_to_old_index.items())

                    def old_to_new(x):
                        if x in list(old_to_new_index.keys()):
                            return old_to_new_index[x]
                        else:
                            return x

                    # perform the reordering
                    new_dag = lang.ExpressionDAG()
                    num_expressions = len(proto.expressions)
                    for cur_index in range(num_expressions):
                        # copy expressions from old spot to new spot
                        cur_expr = proto.expressions[new_to_old(cur_index)]
                        head_expr = new_dag.expressions.add()
                        head_expr.CopyFrom(cur_expr)

                        # copy and update references from old spot to new spot
                        cur_refs = []
                        for ref in proto.references[new_to_old(cur_index)].operand_indices:
                            cur_refs.append(old_to_new(ref))
                        head_reference = new_dag.references.add()
                        head_reference.operand_indices.extend(cur_refs)

                    proto = new_dag

                # finished reordering conditionals, get rid of reordering info
                if_block_start = if_block_start[:-1]
                needs_reordering = needs_reordering[:-1]

        return proto

    @staticmethod
    def from_proto(expression_dag):
        """
        Clear the current ExpressionDAG and build up a fresh one from a serialized protocol buffer.
        :param expression_dag: the serialized protobuf
        :return: None
        """
        code_to_class = {
            lang.INPUT: InputTensor,
            lang.OUTPUT: OutputTensor,
            lang.CONST_SCALAR: _ConstScalar,
            lang.CONST_TENSOR: _ConstTensor,
            lang.POSITION: PositionTensor,
            lang.VARIABLE: Variable,
            lang.CAST: _Cast,
            lang.TENSOR: LocalTensor,
            lang.ASSIGN_VARIABLE: _AssignVariable,
            lang.ASSIGN_TENSOR: _AssignTensor,
            lang.READ_TENSOR: _ReadTensor,
            lang.RANGE: _Range,
            lang.ENDRANGE: _EndRange,
            lang.IF: _If,
            lang.ELSEIF: _ElseIf,
            lang.ELSE: _Else,
            lang.ENDIF: _EndIf,
            lang.ACOS: _UnaryMath,
            lang.ASIN: _UnaryMath,
            lang.ATAN: _UnaryMath,
            lang.COS: _UnaryMath,
            lang.COSH: _UnaryMath,
            lang.SIN: _UnaryMath,
            lang.SINH: _UnaryMath,
            lang.TAN: _UnaryMath,
            lang.TANH: _UnaryMath,
            lang.EXP: _UnaryMath,
            lang.LOG: _UnaryMath,
            lang.LOG10: _UnaryMath,
            lang.SQRT: _UnaryMath,
            lang.CEIL: _UnaryMath,
            lang.FLOOR: _UnaryMath,
            lang.ABS: _UnaryMath,
            lang.NEGATE: _UnaryMath,
            lang.NOT: _UnaryMath,
            lang.ADD: _BinaryMath,
            lang.SUBTRACT: _BinaryMath,
            lang.MULTIPLY: _BinaryMath,
            lang.DIVIDE: _BinaryMath,
            lang.MODULO: _BinaryMath,
            lang.AND: _BinaryMath,
            lang.OR: _BinaryMath,
            lang.EQUAL: _BinaryMath,
            lang.NOTEQUAL: _BinaryMath,
            lang.LESS: _BinaryMath,
            lang.LESS_EQ: _BinaryMath,
            lang.GREATER: _BinaryMath,
            lang.GREATER_EQ: _BinaryMath,
            lang.MIN: _BinaryMath,
            lang.MAX: _BinaryMath,
            lang.POW: _BinaryMath,
            lang.ATAN2: _BinaryMath
        }

        ExpressionDAG.clear()

        # iterate through each proto expression and build up the graph
        for i, expr in enumerate(expression_dag.expressions):
            cur_refs = expression_dag.references[i].operand_indices
            input_exprs = []
            for cur_ref in cur_refs:
                input_exprs.append(ExpressionDAG.exprs[cur_ref])
            code_to_class[expr.code].from_proto(expr, input_exprs)

    @staticmethod
    def generate(expression_dag, function_name):
        """
        Generate C and CUDA code for evaluating the operation defined in the supplied serialized expression dag
        protocol buffer.
        :param expression_dag: The protobuf
        :param function_name: The name of the function to use
        :return: a tuple containing the source for: the individual c function, individual cuda function, the
          standalone cuda function launcher, the generic c++ interface, and the generic cuda interface
        """
        def _strip_margin(s):
            return re.sub('\n[ \t]*\|', '\n', s)
        ExpressionDAG.from_proto(expression_dag)

        inputs = list()
        outputs = list()
        position = None
        for expr in ExpressionDAG.exprs:
            if type(expr) is InputTensor:
                inputs.append(expr)
            elif type(expr) is OutputTensor:
                outputs.append(expr)
            elif type(expr) is PositionTensor:
                position = expr

        num_inputs = len(inputs)
        num_outputs = len(outputs)

        args = list()
        for arg in inputs + outputs:
            args.append(arg.gen_ptr())

        args_str = _list_to_str(args)
        workgroup_shape = position.proto_expr.uint32_data
        workgroup_block_size = [1]
        num_workers = 1
        for cur_dim in workgroup_shape:
            num_workers *= cur_dim

        expression_src = ''
        for expr in ExpressionDAG.exprs:
            try:
                cur_c = expr.gen_c()
            except NotImplementedError as e:
                raise NotImplementedError(str(expr))
            if cur_c != '':
                expression_src += '        ' + cur_c

        # generate c function
        c_src = """
        |//Generated Code
        |#include <stdint.h>
        |#include <stdlib.h>
        |#include <math.h>
        |
        |//aliases for integer absolute values for ints < 32 bits in size
        |//aliases are used because abs() does not work with for
        |// 8 and 16 bit ints in CUDA. We use an alias here so that we can share
        |// code generation infrastructure.
        |#define abs_8(x) abs(x);
        |#define abs_16(x) abs(x);
        |
        |void ${function_name}(${args_str},
        |                      uint32_t block_size, uint32_t thread_index){
        |    uint32_t start = thread_index * block_size;
        |    uint32_t end = start + block_size;
        |    if (end > ${num_workers}) end = ${num_workers};
        |    for(uint32_t worker_index=start; worker_index < end; worker_index++){
        |${expression_src}
        |    }
        |}
        |"""
        c_src = string.Template(c_src).substitute(locals())
        c_src = _strip_margin(c_src)

        # Generate cuda function
        # TODO: make sure that these typedefs are consistent at runtime?
        cuda_defs = _strip_margin(string.Template("""
        |typedef char int8_t;
        |typedef short int16_t;
        |typedef int int32_t;
        |typedef long int64_t;
        |typedef unsigned char uint8_t;
        |typedef unsigned short uint16_t;
        |typedef unsigned int uint32_t;
        |typedef unsigned long uint64_t;
        """).substitute(locals()))

        cuda_function = _strip_margin(string.Template("""
        |//Generated Code
        |
        |//define integer absolute value function
        |inline __device__ int8_t abs_8(const int8_t  & x){ return ( x<0 ) ? -x : x;}
        |inline __device__ int16_t abs_16(const int16_t  & x){ return ( x<0 ) ? -x : x;}
        |
        |extern \"C\" __global__
        |void ${function_name}(${args_str}){
        |    uint32_t worker_index = blockIdx.x * blockDim.x + threadIdx.x;
        |    if (worker_index < ${num_workers}) {
        |${expression_src}
        |    }
        |}
        """).substitute(locals()))

        cuda_src = cuda_defs+cuda_function

        # Generate cuda launcher interface for executing cuda kernels as c function call
        # note that this is slow since it initiates a new CUDA context and is generally only useful for testing
        # allocate_and_copy = ''
        # copy_and_free = ''
        # device_ptrs = []
        # for inp in inputs:
        #     host_ptr = 'in'+str(inp.proto_expr.io_index)
        #     device_ptr = 'd_'+host_ptr
        #     device_ptrs.append(device_ptr)
        #     tipe = inp.dtype.as_cstr()
        #     elements = inp.size
        #     cur_alloc = """
        #     |    size_t ${host_ptr}_size = ${elements}*sizeof(${tipe});
        #     |    CUDA_SAFE_CALL(cuMemAlloc(&${device_ptr}, ${host_ptr}_size));
        #     |    CUDA_SAFE_CALL(cuMemcpyHtoD(${device_ptr}, ${host_ptr}, ${host_ptr}_size));
        #     |"""
        #     allocate_and_copy += string.Template(cur_alloc).substitute(locals())
        #
        #     cur_free = """
        #     |    CUDA_SAFE_CALL(cuMemFree(${device_ptr}));
        #     """
        #     copy_and_free += string.Template(cur_free).substitute(locals())
        #
        # for outp in outputs:
        #     host_ptr = 'out'+str(outp.proto_expr.io_index)
        #     device_ptr = 'd_'+host_ptr
        #     device_ptrs.append(device_ptr)
        #
        #     tipe = outp.dtype.as_cstr()
        #     elements = outp.size
        #     cur_alloc = """
        #     |    size_t ${host_ptr}_size = ${elements}*sizeof(${tipe});
        #     |    CUDA_SAFE_CALL(cuMemAlloc(&${device_ptr}, ${host_ptr}_size));
        #     """
        #     allocate_and_copy += string.Template(cur_alloc).substitute(locals())
        #
        #     cur_free = """
        #     |    CUDA_SAFE_CALL(cuMemcpyDtoH(${host_ptr}, ${device_ptr}, ${host_ptr}_size));
        #     |    CUDA_SAFE_CALL(cuMemFree(${device_ptr}));
        #     """
        #     copy_and_free += string.Template(cur_free).substitute(locals())
        #
        # device_ptrs_string = _list_to_str(device_ptrs)
        #
        # device_args = []
        # for ptr in device_ptrs:
        #     device_args.append('&'+ptr)
        # device_args_string = _list_to_str(device_args)
        #
        # ptx_placeholder = '${ptx_string}'
        # cuda_launch_template = """
        # |//Generated Code
        # |//modified version of NVIDIA's 'SAXPY' example for running ptx compiled by nvrtc:
        # |// http://docs.nvidia.com/cuda/nvrtc/index.html#example-saxpy
        # |
        # |#include <cuda.h>
        # |#include <iostream>
        # |#include <stdint.h>
        # |
        # |#define CUDA_SAFE_CALL(x)                                         \\
        # |    do {                                                          \\
        # |        CUresult result = x;                                      \\
        # |        if (result != CUDA_SUCCESS) {                             \\
        # |            const char *msg;                                      \\
        # |            cuGetErrorName(result, &msg);                         \\
        # |            std::cerr << "\\nerror: " #x " failed with error "     \\
        # |                      << msg << '\\n';                             \\
        # |            exit(1);                                              \\
        # |        }                                                         \\
        # |    } while(0)
        # |
        # |extern "C" uint16_t ${function_name}(${args_str}, uint16_t threads_per_block)
        # |{
        # |    //begin previously compiled ptx string
        # |    const char * ptx = R"(${ptx_placeholder})";
        # |    //end previously compiled ptx string
        # |
        # |    //obtain function handle
        # |    CUdevice cuDevice;
        # |    CUcontext context;
        # |    CUmodule module;
        # |    CUfunction kernel;
        # |    CUDA_SAFE_CALL(cuInit(0));
        # |    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
        # |    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
        # |    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
        # |    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "${function_name}"));
        # |
        # |    //allocate memory on and copy inputs to the device
        # |    CUdeviceptr ${device_ptrs_string};
        # |
        # ${allocate_and_copy}
        # |
        # |    uint32_t num_blocks = ${num_workers} / threads_per_block;
        # |    if(${num_workers} % threads_per_block > 0){ num_blocks += 1;}
        # |
        # |    void *args[] = {${device_args_string}};
        # |    CUDA_SAFE_CALL(
        # |    cuLaunchKernel(kernel,
        # |                   num_blocks, 1, 1,   // grid dim
        # |                   threads_per_block, 1, 1,    // block dim
        # |                   0, NULL,             // shared mem and stream
        # |                   args, 0));           // arguments
        # |    CUDA_SAFE_CALL(cuCtxSynchronize());
        # |
        # |    // copy outputs to host and free device memory
        # ${copy_and_free}
        # |
        # |    CUDA_SAFE_CALL(cuModuleUnload(module));
        # |    CUDA_SAFE_CALL(cuCtxDestroy(context));
        # |
        # |    return 0;
        # |}
        # |"""
        # cuda_launch_template = string.Template(cuda_launch_template).substitute(locals())
        # cuda_launch_template = _strip_margin(cuda_launch_template)

        # Generate the c generic parameter interface for unpacking polymorphic io parameters
        generic_args = []
        io_ptrs = ''
        for inp in inputs:
            cur_index = inp.proto_expr.io_index
            cur_name = 'in'+str(cur_index)
            generic_args.append(cur_name + '.p_fixed_len')
            elements = inp.size
            tipe = inp.dtype.as_cstr()

            io_ptrs += string.Template("""
                |    if(inputs[${cur_index}]->length() != ${elements}) { *err = 1; return; }
                |    union u_in${cur_index}{
                |       const ${tipe} *p_arb_len;
                |       const ${tipe} (*p_fixed_len)[${elements}];
                |    };
                |    union u_in${cur_index} in${cur_index};
                |    in${cur_index}.p_arb_len = inputs[${cur_index}]->get<${tipe}>();
                |""").substitute(locals())

        for outp in outputs:
            cur_index = outp.proto_expr.io_index
            cur_name = 'out'+str(cur_index)
            generic_args.append(cur_name + '.p_fixed_len')
            elements = outp.size
            tipe = outp.dtype.as_cstr()

            io_ptrs += string.Template("""
                |    if(outputs[${cur_index}]->length() != ${elements}) { *err = 1; return; }
                |    union u_out${cur_index}{
                |       ${tipe} *p_arb_len;
                |       ${tipe} (*p_fixed_len)[${elements}];
                |    };
                |    union u_out${cur_index} out${cur_index};
                |    out${cur_index}.p_arb_len = outputs[${cur_index}]->get<${tipe}>();
                |""").substitute(locals())

        args = _list_to_str(generic_args)
        c_generic = """
        |#include "dynamiclibop.h"
        |#include <vector>
        |#include <memory>
        |
        |${c_src}
        |
        |extern "C"
        |void ${function_name}_generic_cpp(std::vector<std::shared_ptr<const InputParameter>> inputs,
        |                                  std::vector<std::shared_ptr<OutputParameter>> outputs,
        |                                  uint32_t num_threads, uint32_t thread_index, uint16_t* err){
        |    //check that the number of inputs and outputs is correct
        |    if(inputs.size() != ${num_inputs}){ *err = 1; return; }
        |    if(outputs.size() != ${num_outputs}){ *err = 1; return; }
        |
        |    //check that the size of inputs and outputs is correct, and cast them as pointers to arrays
        ${io_ptrs}
        |    uint32_t block_size = ${num_workers} / num_threads;
        |    if(${num_workers} % num_threads > 0) block_size += 1;
        |    return ${function_name}(${args}, block_size, thread_index);
        |}
        |"""
        c_generic = string.Template(c_generic).substitute(locals())
        c_generic = _strip_margin(c_generic)

        cuda_generic = """
        |#include "dynamiclibop.h"
        |#include <vector>
        |#include <string>
        |#include <memory>
        |#include <cuda.h>
        |
        |${cuda_function}
        |
        |extern "C"
        |void ${function_name}_generic_cuda(std::vector<std::shared_ptr<const InputParameter>> inputs,
        |                                   std::vector<std::shared_ptr<OutputParameter>> outputs,
        |                                   CUstream stream, uint16_t threads_per_block, uint16_t* err){
        |    //check that the number of inputs and outputs is correct
        |    if(inputs.size() != ${num_inputs}){ *err = 1; return; }
        |    if(outputs.size() != ${num_outputs}){ *err = 1; return; }
        |
        |    //check that the size of inputs and outputs is correct, and cast them as pointers to arrays
        ${io_ptrs}
        |    //enqueue function on stream
        |    uint32_t num_blocks = ${num_workers} / threads_per_block;
        |    if(${num_workers} % threads_per_block > 0) num_blocks += 1;
        |    ${function_name}<<<num_blocks, threads_per_block, 0, stream>>>(${args});
        |}
        """
        cuda_generic = string.Template(cuda_generic).substitute(locals())
        cuda_generic = _strip_margin(cuda_generic)

        # Generate the cuda generic parameter interface for unpacking polymorphic io parameters

        return c_src, cuda_src, c_generic, cuda_generic

    @staticmethod
    def expr_index(expr):
        """
        Resolve the index of a particular expression in the current DAG
        :param expr: the expression
        :return: its index
        """
        return ExpressionDAG.expr_ids.index(id(expr))


class _Expression(object):
    """
    The abstract class that defines the behavior of expressions.
    """
    def __init__(self, expression_code):
        # assign a protocol buffers member corresponding to this python object
        if expression_code not in list(lang.ExpressionCode.values()):
            raise ValueError('Expression code ' + str(expression_code) + ' is not valid.')
        self.proto_expr = lang.Expression()
        self.proto_expr.code = expression_code
        self.input_exprs = []
        self.name = None

    def _register(self):
        ExpressionDAG.append(self)

    def __str__(self):
        return str(self.proto_expr)

    def gen_c(self):
        raise NotImplementedError('Abstract Class')

    def __ilshift__(self, other):
        raise SyntaxError('Can only use assignment operator <<= on a variable.')

    @staticmethod
    def from_proto(proto, input_exprs):
        raise NotImplementedError('Abstract Class')

    def __bool__(self):
        self.__nonzero__()

    def __nonzero__(self):
        raise SyntaxError('Attempting to interpret the truth of an expression. This typically happens when trying to '
                          'use a python native "if", "min", or "max" statement to create a data-dependent conditional '
                          'inside of an operator, which is not supported. To do so you must use the corresponding '
                          '"with if_(...)", "minimum", and "maximum" functions.')


class Scalar(_Expression):
    """
    An expression that refers to a single data value which has a data type
    """
    def __init__(self, expr_code, dtype):
        if not isinstance(dtype, DType):
            raise TypeError('Scalar expressions must be initialized with a DType')
        super(Scalar, self).__init__(expr_code)
        self.dtype = dtype
        self.proto_expr.dtype = dtype.as_proto()

    def __add__(self, other):
        return _BinaryMath(self, other, lang.ADD)

    def __radd__(self, other):
        return _BinaryMath(other, self, lang.ADD)

    def __sub__(self, other):
        return _BinaryMath(self, other, lang.SUBTRACT)

    def __rsub__(self, other):
        return _BinaryMath(other, self, lang.SUBTRACT)

    def __mul__(self, other):
        return _BinaryMath(self, other, lang.MULTIPLY)

    def __rmul__(self, other):
        return _BinaryMath(other, self, lang.MULTIPLY)

    # python 2
    def __div__(self, other):
        return _BinaryMath(self, other, lang.DIVIDE)

    def __rdiv__(self, other):
        return _BinaryMath(other, self, lang.DIVIDE)

    # python 3
    def __truediv__(self, other):
        return _BinaryMath(self, other, lang.DIVIDE)

    def __rtruediv__(self, other):
        return _BinaryMath(other, self, lang.DIVIDE)

    def __mod__(self, other):
        return _BinaryMath(self, other, lang.MODULO)

    def __rmod__(self, other):
        return _BinaryMath(other, self, lang.MODULO)

    def __eq__(self, other):
        return _BinaryMath(self, other, lang.EQUAL)

    def __ne__(self, other):
        return _BinaryMath(self, other, lang.NOTEQUAL)

    def __lt__(self, other):
        return _BinaryMath(self, other, lang.LESS)

    def __le__(self, other):
        return _BinaryMath(self, other, lang.LESS_EQ)

    def __gt__(self, other):
        return _BinaryMath(self, other, lang.GREATER)

    def __ge__(self, other):
        return _BinaryMath(self, other, lang.GREATER_EQ)

    def __neg__(self):
        return _UnaryMath(self, lang.NEGATE)

    @staticmethod
    def from_proto(proto, input_exprs):
        raise NotImplementedError('Abstract Class')

    def gen_c(self):
        raise NotImplementedError('Abstract Class')


class _TensorExpression(_Expression):
    """
    An expression that refers to a tensor of data which has a TensorType
    """
    def __init__(self, expr_code, tensor_type):
        super(_TensorExpression, self).__init__(expr_code)
        self.dtype = tensor_type.dtype
        self.shape = tensor_type.shape
        self.size = tensor_type.size
        self.rank = tensor_type.rank
        self.tensor_type = tensor_type
        self.proto_expr.tensor_type.CopyFrom(tensor_type.as_proto())

    def gen_ptr(self):
        raise NotImplementedError('Abstract Class')

    @staticmethod
    def from_proto(proto, input_exprs):
        raise NotImplementedError('Abstract Class')

    def gen_c(self):
        raise NotImplementedError('Abstract Class')


def _to_scalar_index(target_shape, index):
    """
    Helper function for indexing tensors. All tensors are stored as C-style flattened arrays in memory, but are indexed
     from the API with an index for each dimension. This function resolves the scalar index of the tensor memory from
     the input index. The length of the input index must always be the same as the rank of the target tensor. Indices
     can be a tensor, or a mixed iterable of constants, scalar expressions, and 0-D tensor expressions.
    :param target: The tensor to be indexed
    :param index: The index tensor or iterable
    :return: a scalar expression containing the index of the flattened target tensor memory
    """
    target_rank = len(target_shape)
    block_size = [1]
    for cur_dim in range(len(target_shape)-1, 0, -1):
        block_size.append(block_size[-1]*target_shape[cur_dim])
    block_size.reverse()

    # try to wrap index as a const tensor
    try:
        index_expr = _ConstTensor(index)
    except TypeError:
        index_expr = index

    # wrap scalar expressions as lists
    if issubclass(index_expr.__class__, Scalar):
        index_expr = [index_expr]

    # try to wrap index as an explicit array tensor
    # (e.g. img[row, column] where both row and column are scalar variables of the same type)
    # allow for constants to be mixed with expressions (e.g. img[row, 5])
    if type(index_expr) is list or type(index_expr) is tuple:
        explicit_len = len(index_expr)
        if target_rank != explicit_len:
            raise IndexError('length of index list (' + str(explicit_len) +
                             ') must match indexed tensor rank (' + str(target_rank) + ')')

        exprs = []
        for value in index_expr:
            if issubclass(value.__class__, _Expression):
                if not issubclass(value.__class__, Scalar):
                    if issubclass(value.__class__, _TensorExpression) and value.size == 1:
                        # enable indexing with size == 1 tensor
                        # This typically arises when the workground shape is 1D and the position
                        # tensor is a single value.
                        value = value[0]
                    else:
                        raise TypeError('Must index tensors with an int or a scalar expression. Instead got:\n' +
                                        str(value))
                exprs.append(value)
            else:
                # this dimension is constant, perform static bounds checking
                cur_dim = len(exprs)
                cur_shape = target_shape[cur_dim]
                cur_value = int(np.floor(value))
                if cur_value >= cur_shape or cur_value < 0:
                    raise IndexError('Expected index to be in range [0, ' + str(cur_shape) +
                                     '), but received index value ' + str(cur_value))

                exprs.append(cur_value)

        index = None
        for i, expr in enumerate(exprs):
            if not isinstance(expr, int):
                # todo: optionally dynamically constrain each non-constant dimensional index to within shape bounds
                # bound_expr = cast(minimum(maximum(expr, 0), target_shape[i]-1), uint64)
                bound_expr = cast(expr, uint64)
            else:
                bound_expr = expr

            if index is None:
                index = bound_expr*block_size[i]
            else:
                index = index + bound_expr*block_size[i]
        return index

    elif type(index_expr) is _ConstTensor:
        # indexing with a constant, perform static bounds checking
        if len(index_expr.shape) != 1:
            raise IndexError('Index must be one dimensional')
        if index_expr.shape[0] != target_rank:
            raise IndexError('length of index tensor (' + str(index_expr.shape[0]) +
                             ') must match indexed tensor rank (' + str(target_rank) + ')')

        data = np.floor(index_expr.to_array())
        index = 0
        for i, elem in enumerate(data):
            cur_shape = target_shape[i]
            cur_value = int(elem)
            if cur_value >= cur_shape or cur_value < 0:
                raise IndexError('Expected index to be in range [0, ' + str(cur_shape) +
                                 '), but received index value ' + str(cur_value))
            index += int(elem)*block_size[i]

        return index
    elif type(index_expr) in [LocalTensor, PositionTensor, InputTensor]:
        if len(index_expr.shape) != 1:
            raise IndexError('Index must be one dimensional')
        if index_expr.shape[0] != target_rank:
            raise IndexError('length of index tensor (' + str(index_expr.shape[0]) +
                             ') must match indexed tensor rank (' + str(target_rank) + ')')

        index = None
        for i in range(target_rank):
            cur_shape = target_shape[i]
            cur_index = _ReadTensor(index_expr, i)
            # todo: optionally dynamically constrain each dimensional index to within shape bounds
            # bound_index = minimum(maximum(cast(cur_index, uint64), 0), cur_shape-1)
            bound_index = cast(cur_index, uint64)
            if index is None:
                index = bound_index*block_size[i]
            else:
                index = index + bound_index*block_size[i]

        return index
    else:
        raise TypeError('Cannot index a tensor with a ' + str(type(index_expr)))


class _Readable(object):
    """
    A trait for tensors which enables them to be read
    """
    def __getitem__(self, item):
        return _ReadTensor(self, _to_scalar_index(self.shape, item))


class _Writable(object):
    """
    A trait for tensors which enables them to be written
    """
    def __setitem__(self, key, value):
        _AssignTensor(self, _to_scalar_index(self.shape, key), value)


def _tensor_type_polymorhpic(*args):
    """
    A helper function for resolving polymorphic inputs into a TensorType
    :param args: args the define a TensorType, can be either a TensorType or a shape and a DType
    :return: the resolved TensorType
    """
    err_msg = 'Expected a TensorType or a shape and a dtype as arguments'
    if len(args) == 1:
        if type(args[0]) is not TensorType:
            raise TypeError(err_msg)
        tensor_type = args[0]
    elif len(args) == 2:
        tensor_type = TensorType(args[0], args[1])
    else:
        raise TypeError(err_msg)

    return tensor_type


def input(*args):
    """
    Create a new input
    :param args: args the define a TensorType, can be either a TensorType or a shape and a DType
    :return: the input expression
    """
    tensor_type = _tensor_type_polymorhpic(*args)
    return InputTensor(tensor_type, ExpressionDAG.num_inputs)


class InputTensor(_TensorExpression, _Readable):
    """
    A read-only input tensor expression
    """
    def __init__(self, tensor_type, io_index):
        if not isinstance(tensor_type, TensorType):
            raise TypeError
        if not isinstance(io_index, int):
            raise TypeError

        super(self.__class__, self).__init__(lang.INPUT, tensor_type)

        if io_index < 0 or io_index > 2**32-1:
            raise ValueError
        self.proto_expr.io_index = io_index

        super(self.__class__, self)._register()

    def gen_ptr(self):
        tipe = self.dtype.as_cstr()
        name = self.name
        elems = self.size
        p = string.Template('const ${tipe} ${name}[${elems}]').substitute(locals())

        return p

    @staticmethod
    def from_proto(proto, input_exprs):
        tt = TensorType.from_proto(proto.tensor_type)
        return InputTensor(tt, proto.io_index)

    def gen_c(self):
        return ''


def output(*args):
    """
    Define a new output

    :param args: args the define a TensorType, can be either a TensorType or a shape and a DType
    :return: a tensor expression which refers to the newly defined output tensor

    :Example:

    Create a new output tensor ``out`` based on the ``TensorType`` of input tensor ``in0`` ::

        out = output(in0.tensor_type)

    :Example:

    Create a new output tensor ``out`` based on the ``shape`` of input tensor ``in0`` and the ``DType`` of input tensor
    ``in1``::

        out = output(in0.shape, in1.dtype)

    """

    tensor_type = _tensor_type_polymorhpic(*args)
    return OutputTensor(tensor_type, ExpressionDAG.num_outputs)


def output_like(other):
    """
    Define a new output with the same TensorType as another tensor

    :param other: another tensor
    :return: a tensor expression which refers to the newly defined output tensor
    """

    return output(TensorType.like(other))


class OutputTensor(_TensorExpression, _Writable):
    """
    A write-only output expression
    """
    def __init__(self, tensor_type, io_index):
        if not isinstance(tensor_type, TensorType):
            raise TypeError
        if not isinstance(io_index, int):
            raise TypeError

        super(self.__class__, self).__init__(lang.OUTPUT, tensor_type)

        if io_index < 0 or io_index > 2**32-1:
            raise ValueError
        self.proto_expr.io_index = io_index

        super(self.__class__, self)._register()

    def gen_ptr(self):
        tipe = self.dtype.as_cstr()
        name = self.name
        elems = self.size
        p = string.Template('${tipe} ${name}[${elems}]').substitute(locals())

        return p

    @staticmethod
    def from_proto(proto, input_exprs):
        tt = TensorType.from_proto(proto.tensor_type)
        return OutputTensor(tt, proto.io_index)

    def gen_c(self):
        return ''


class _ConstScalar(Scalar):
    """
    A constant expression
    """
    def __init__(self, value):
        if type(value) is float:
            super(self.__class__, self).__init__(lang.CONST_SCALAR, float64)
            self.proto_expr.double_data.append(value)
        elif type(value) is int:
            super(self.__class__, self).__init__(lang.CONST_SCALAR, int64)
            self.proto_expr.sint64_data.append(value)
        else:
            tipe = str(type(value))
            raise TypeError('Tried to wrap a '+tipe+' as a ConstScalar. Can only wrap an int or float')

        super(self.__class__, self)._register()

    def value(self):
        if self.proto_expr.dtype == lang.FLOAT64:
            return float(self.proto_expr.double_data[0])
        elif self.proto_expr.dtype == lang.INT64:
            return int(self.proto_expr.sint64_data[0])
        else:
            raise ValueError('Can only get a value from float64 or int64 constants.')

    @staticmethod
    def from_proto(proto, input_exprs):
        if proto.dtype == lang.FLOAT64:
            return _ConstScalar(float(proto.double_data[0]))
        elif proto.dtype == lang.FLOAT32:
            return _ConstScalar(float(proto.float_data[0]))
        elif proto.dtype == lang.INT64:
            return _ConstScalar(int(proto.sint64_data[0]))
        else:
            raise ValueError('Cannot recover constant scalar protobuf.')

    def gen_c(self):
        # return 'const ' + self.dtype.as_cstr() + ' ' + self.name + ' = ' + str(self.value()) + ';\n'
        return ''


class _ConstTensor(_TensorExpression, _Readable):
    """
    A constant tensor expression
    """

    # translation table between dtypes and retrieval function for the data container to use
    proto_data_lut = {
        float16: lambda x: x.float_data,
        float32: lambda x: x.float_data,
        float64: lambda x: x.double_data,
        int8: lambda x: x.sint32_data,
        int16: lambda x: x.sint32_data,
        int32: lambda x: x.sint32_data,
        int64: lambda x: x.sint64_data,
        uint8: lambda x: x.uint32_data,
        uint16: lambda x: x.uint32_data,
        uint32: lambda x: x.uint32_data,
        uint64: lambda x: x.uint64_data
    }

    def __init__(self, value):

        # use numpy functionality to convert lists and tuples to arrays
        if type(value) is list:
            array = np.array(value)
        elif type(value) is tuple:
            array = np.array(value)
        elif type(value) is np.ndarray:
            array = value
        elif type(value) is int or type(value) is float:
            array = np.array([value])
        else:
            raise TypeError('ConstTensors can wrap lists, tuples, and numpy arrays')

        super(self.__class__, self).__init__(lang.CONST_TENSOR, TensorType.like(array))

        # build up protobuf representation
        flat_data = array.flatten(order='C').tolist()
        vals = list(_ConstTensor.proto_data_lut.values())
        keys = list(_ConstTensor.proto_data_lut.keys())
        proto_data_retrieval = vals[keys.index(self.tensor_type.dtype)]
        proto_data = proto_data_retrieval(self.proto_expr)
        proto_data.extend(flat_data)

        super(self.__class__, self)._register()

    def to_array(self):
        vals = list(_ConstTensor.proto_data_lut.values())
        keys = list(_ConstTensor.proto_data_lut.keys())
        proto_data_retrieval = vals[keys.index(self.tensor_type.dtype)]
        proto_data = proto_data_retrieval(self.proto_expr)
        data = np.array(proto_data, dtype=self.dtype.as_numpy())
        return data

    @staticmethod
    def from_proto(proto, input_exprs):
        dtype = DType(proto.tensor_type.dtype)

        vals = list(_ConstTensor.proto_data_lut.values())
        keys = list(_ConstTensor.proto_data_lut.keys())
        proto_data_retrieval = vals[keys.index(dtype)]
        proto_data = proto_data_retrieval(proto)
        data = np.array(proto_data, dtype=dtype.as_numpy())
        return _ConstTensor(data)

    def gen_ptr(self):
        tipe = self.dtype.as_cstr()
        name = self.name
        elems = self.size

        return string.Template('const ${tipe} ${name}[${elems}]').substitute(locals())

    def gen_c(self):
        return ''


def position_in(workgroup_shape):
    """
    Define the workgroup shape and retrieve a tensor expression that refers to the current position in that
    workgroup shape.

    :param workgroup_shape: An iterable of ints defining the shape of the workgroup
    :return: a tensor expression which references the current workgroup position
    """
    return PositionTensor(workgroup_shape)


class PositionTensor(_TensorExpression, _Readable):
    """
    The position expression which refers to the current position within the workgroup shape
    """
    def __init__(self, workgroup_shape):

        if type(workgroup_shape) is int:
            self.workgroup_shape = [workgroup_shape]
        else:
            try:
                for elem in workgroup_shape:
                    if not isinstance(elem, int):
                        raise TypeError
            except TypeError:
                raise TypeError('workgroup_shape must be an int or an iterable of ints')

            self.workgroup_shape = workgroup_shape

        workgroup_dims = len(self.workgroup_shape)
        tensor_type = TensorType([workgroup_dims], uint32)

        super(self.__class__, self).__init__(lang.POSITION, tensor_type)

        self.proto_expr.uint32_data.extend(self.workgroup_shape)

        super(self.__class__, self)._register()

    @staticmethod
    def from_proto(proto, input_exprs):
        return PositionTensor(proto.uint32_data)

    def gen_ptr(self):
        tipe = self.dtype.as_cstr()
        name = self.name
        elems = self.size
        p = string.Template('${tipe} ${name}[${elems}]').substitute(locals())

    def gen_c(self):
        workgroup_block_size = [1]
        for cur_dim in range(self.size-1, 0, -1):
            workgroup_block_size.append(workgroup_block_size[-1]*self.workgroup_shape[cur_dim])
        workgroup_block_size.reverse()

        position_vals = []
        remainder = 'worker_index'
        for cur_block in workgroup_block_size:
            cur_index = '('+remainder+')/'+str(cur_block)
            position_vals.append(cur_index)
            remainder = remainder + ' % ' + str(cur_block)

        return 'const uint32_t position['+str(self.size)+'] = {' + _list_to_str(position_vals) + '};\n'


def variable(initial_value, dtype):
    """
    Function for declaring a new variable

    :param initial_value: The initial value of the variable
    :param dtype: The DType of the variable
    :return: The variable expression
    """
    if type(initial_value) is int or type(initial_value) is float:
        return Variable(dtype, _ConstScalar(initial_value))
    elif issubclass(initial_value.__class__, Scalar):
        var = Variable(dtype, _ConstScalar(0))
        var <<= initial_value
        return var
    else:
        raise TypeError('Must initialize a variable with a numeric constant or a scalar expression.')


class Variable(Scalar):
    """
    A variable expression
    """
    def __init__(self, dtype, intial_const):
        if not isinstance(intial_const, _ConstScalar):
            raise TypeError('Variables must be initialized with a constant scalar')
        if not isinstance(dtype, DType):
            raise TypeError('dtype must be a DType')

        super(self.__class__, self).__init__(lang.VARIABLE, dtype)

        self.input_exprs = [intial_const]

        super(self.__class__, self)._register()

    def __ilshift__(self, other):
        _AssignVariable(self, other)
        return self

    @staticmethod
    def from_proto(proto, input_exprs):
        return Variable(DType(proto.dtype), input_exprs[0])

    def gen_c(self):
        return self.dtype.as_cstr() + ' ' + self.name + ' = ' + self.input_exprs[0].name + ';\n'


def cast(value, dtype):
    """
    Cast a scalar expression as a new data type

    :param value: The scalar expression
    :param dtype: The new data type
    :return: The casted scalar expression
    """
    return _Cast(dtype, value)


class _Cast(Scalar):
    """
    The casting expression
    """
    def __init__(self, dtype, target):
        if not isinstance(dtype, DType):
            raise TypeError('dtype must be a DType')
        if not issubclass(target.__class__, Scalar):
            raise TypeError('Can only cast scalar expressions. Received ' + str(type(target)) + ': ' +
                            str(target))

        super(self.__class__, self).__init__(lang.CAST, dtype)
        self.input_exprs = [target]

        super(self.__class__, self)._register()

    @staticmethod
    def from_proto(proto, input_exprs):
        return _Cast(DType(proto.dtype), input_exprs[0])

    def gen_c(self):
        return self.dtype.as_cstr() + ' ' + self.name + ' = ' + self.input_exprs[0].name + ';\n'


class _AssignVariable(_Expression):
    """
    The variable assignment expression
    """
    def __init__(self, scalar_expr, value_expr):
        if not isinstance(scalar_expr, Variable):
            raise TypeError('Can only assign to a variable')

        if issubclass(value_expr.__class__, Scalar):
            value = value_expr
        else:
            value = _ConstScalar(value_expr)
            value = cast(value, scalar_expr.dtype)

        super(self.__class__, self).__init__(lang.ASSIGN_VARIABLE)

        t1 = scalar_expr.proto_expr.dtype
        t2 = value.proto_expr.dtype

        if not t1 == t2:
            t1_str = lang.DType.Name(t1)
            t2_str = lang.DType.Name(t2)
            raise TypeError('cannot assign ' + t2_str + ' to ' + t1_str + ' scalar')

        self.input_exprs = [scalar_expr, value]

        super(self.__class__, self)._register()

    @staticmethod
    def from_proto(proto, input_exprs):
        return _AssignVariable(input_exprs[0], input_exprs[1])

    def gen_c(self):
        return self.input_exprs[0].name + ' = ' + self.input_exprs[1].name + ';\n'


class _UnaryMath(Scalar):
    """
    Unary expressions which transform a single scalar expression
    """
    code_map = {
        lang.ACOS: {lang.FLOAT32: 'acosf', lang.FLOAT64: 'acos'},
        lang.ASIN: {lang.FLOAT32: 'asinf', lang.FLOAT64: 'asin'},
        lang.ATAN: {lang.FLOAT32: 'atanf', lang.FLOAT64: 'atan'},
        lang.COS: {lang.FLOAT32: 'cosf', lang.FLOAT64: 'cos'},
        lang.COSH: {lang.FLOAT32: 'coshf', lang.FLOAT64: 'cosh'},
        lang.SIN: {lang.FLOAT32: 'sinf', lang.FLOAT64: 'sin'},
        lang.SINH: {lang.FLOAT32: 'sinhf', lang.FLOAT64: 'sinh'},
        lang.TAN: {lang.FLOAT32: 'tanf', lang.FLOAT64: 'tan'},
        lang.TANH: {lang.FLOAT32: 'tanhf', lang.FLOAT64: 'tanh'},
        lang.EXP: {lang.FLOAT32: 'expf', lang.FLOAT64: 'exp'},
        lang.LOG: {lang.FLOAT32: 'logf', lang.FLOAT64: 'log'},
        lang.LOG10: {lang.FLOAT32: 'log10f', lang.FLOAT64: 'log10'},
        lang.SQRT: {lang.FLOAT32: 'sqrtf', lang.FLOAT64: 'sqrt'},
        lang.CEIL: {lang.FLOAT32: 'ceilf', lang.FLOAT64: 'ceil'},
        lang.FLOOR: {lang.FLOAT32: 'floorf', lang.FLOAT64: 'floor'},
        lang.ABS: {lang.FLOAT32: 'fabsf', lang.FLOAT64: 'fabs', lang.INT8: 'abs_8',
                   lang.INT16: 'abs_16', lang.INT32: 'abs', lang.INT64: 'labs'},
        lang.NEGATE: {lang.FLOAT32: '-', lang.FLOAT64: '-',
                      lang.INT8: '-', lang.INT16: '-', lang.INT32: '-', lang.INT64: '-'},
        lang.NOT: {lang.FLOAT32: '!', lang.FLOAT64: '!',
                      lang.INT8: '!', lang.INT16: '!', lang.INT32: '!', lang.INT64: '!',
                      lang.UINT8: '!', lang.UINT16: '!', lang.UINT32: '!', lang.UINT64: '!'}
    }

    def __init__(self, arg, expr_code):
        if expr_code not in list(_UnaryMath.code_map.keys()):
            raise ValueError(lang.ExpressionCode.Name(expr_code) + 'is an invalid unary math code.')

        if arg.dtype.proto_dtype not in list(_UnaryMath.code_map[expr_code].keys()):
            raise ValueError(lang.DType.Name(arg.dtype.proto_dtype) +
                             ' arguments not supported for unary math function ' +
                             lang.ExpressionCode.Name(expr_code))

        if not issubclass(arg.__class__, Scalar):
            raise TypeError('Must apply math functions to scalar expressions. Received: ' + str(arg))

        super(self.__class__, self).__init__(expr_code, arg.dtype)

        self.input_exprs = [arg]
        super(self.__class__, self)._register()

    @staticmethod
    def from_proto(proto, input_exprs):
        return _UnaryMath(input_exprs[0], proto.code)

    def gen_c(self):
        func_string = _UnaryMath.code_map[self.proto_expr.code][self.proto_expr.dtype]
        return self.dtype.as_cstr() + ' ' + self.name + ' = ' + func_string + '(' + self.input_exprs[0].name + ');\n'


def arccos(x):
    return _UnaryMath(x, lang.ACOS)


def arcsin(x):
    return _UnaryMath(x, lang.ASIN)


def arctan(x):
    return _UnaryMath(x, lang.ATAN)


def cos(x):
    return _UnaryMath(x, lang.COS)


def cosh(x):
    return _UnaryMath(x, lang.COSH)


def sin(x):
    return _UnaryMath(x, lang.SIN)


def sinh(x):
    return _UnaryMath(x, lang.SINH)


def tan(x):
    return _UnaryMath(x, lang.TAN)


def tanh(x):
    return _UnaryMath(x, lang.TANH)


def exp(x):
    return _UnaryMath(x, lang.EXP)


def log(x):
    return _UnaryMath(x, lang.LOG)


def log10(x):
    return _UnaryMath(x, lang.LOG10)


def sqrt(x):
    return _UnaryMath(x, lang.SQRT)


def ceil(x):
    return _UnaryMath(x, lang.CEIL)


def floor(x):
    return _UnaryMath(x, lang.FLOOR)


def absolute(x):
    return _UnaryMath(x, lang.ABS)


def logical_not(x):
    return _UnaryMath(x, lang.NOT)


class _BinaryMath(Scalar):
    """
    Binary expressions which transform two scalars into another
    """
    code_map = {
        lang.ADD: {},
        lang.SUBTRACT: {},
        lang.MULTIPLY: {},
        lang.DIVIDE: {},
        lang.MODULO: {},
        lang.EQUAL: {},
        lang.NOTEQUAL: {},
        lang.LESS: {},
        lang.LESS_EQ: {},
        lang.GREATER: {},
        lang.GREATER_EQ: {},
        lang.MIN: {},
        lang.MAX: {},
        lang.AND: {},
        lang.OR: {},
        lang.POW: {lang.FLOAT32: lambda x, y: 'powf('+x+','+y+')',
                   lang.FLOAT64: lambda x, y: 'pow('+x+','+y+')'},
        lang.ATAN2: {lang.FLOAT32: lambda x, y: 'atan2f('+x+','+y+')',
                     lang.FLOAT64: lambda x, y: 'atan2('+x+','+y+')'},
    }

    for cur_type in supported_types:
        code_map[lang.MIN][cur_type.proto_dtype] = lambda x, y: '((('+x+')<('+y+'))?('+x+'):('+y+'))'
        code_map[lang.MAX][cur_type.proto_dtype] = lambda x, y: '((('+x+')>('+y+'))?('+x+'):('+y+'))'
        code_map[lang.ADD][cur_type.proto_dtype] = lambda x, y: '(' + x + ' + ' + y + ')'
        code_map[lang.SUBTRACT][cur_type.proto_dtype] = lambda x, y: '(' + x + ' - ' + y + ')'
        code_map[lang.MULTIPLY][cur_type.proto_dtype] = lambda x, y: '(' + x + ' * ' + y + ')'
        code_map[lang.DIVIDE][cur_type.proto_dtype] = lambda x, y: '(' + x + ' / ' + y + ')'
        code_map[lang.MODULO][cur_type.proto_dtype] = lambda x, y: '(' + x + ' % ' + y + ')'
        code_map[lang.EQUAL][cur_type.proto_dtype] = lambda x, y: '(' + x + ' == ' + y + ')'
        code_map[lang.NOTEQUAL][cur_type.proto_dtype] = lambda x, y: '(' + x + ' != ' + y + ')'
        code_map[lang.LESS][cur_type.proto_dtype] = lambda x, y: '(' + x + ' < ' + y + ')'
        code_map[lang.LESS_EQ][cur_type.proto_dtype] = lambda x, y: '(' + x + ' <= ' + y + ')'
        code_map[lang.GREATER][cur_type.proto_dtype] = lambda x, y: '(' + x + ' > ' + y + ')'
        code_map[lang.GREATER_EQ][cur_type.proto_dtype] = lambda x, y: '(' + x + ' >= ' + y + ')'
        code_map[lang.AND][cur_type.proto_dtype] = lambda x, y: '(' + x + ' && ' + y + ')'
        code_map[lang.OR][cur_type.proto_dtype] = lambda x, y: '(' + x + ' || ' + y + ')'

    code_map[lang.MODULO][float32.proto_dtype] = lambda x, y: 'fmodf('+x+','+y+')'
    code_map[lang.MODULO][float64.proto_dtype] = lambda x, y: 'fmod('+x+','+y+')'

    def __init__(self, arg1, arg2, expr_code):
        if expr_code not in list(_BinaryMath.code_map.keys()):
            raise ValueError('Invalid binary math code')
        code_str = lang.ExpressionCode.Name(expr_code)

        # first try to wrap args as constants
        try:
            arg1_wrapped = _ConstScalar(arg1)
        except TypeError:
            arg1_wrapped = arg1

        try:
            arg2_wrapped = _ConstScalar(arg2)
        except TypeError:
            arg2_wrapped = arg2

        # throw error if received a non-expression that could not be wrapped as constant
        if not issubclass(arg1_wrapped.__class__, _Expression):
            raise TypeError('Cannot apply ' + code_str + ' to first non-expression argument:\n' + str(arg1_wrapped))
        if not issubclass(arg2_wrapped.__class__, _Expression):
            raise TypeError('Cannot apply ' + code_str + ' to second non-expression argument:\n' + str(arg2_wrapped))

        # throw error if received a non-scalar expression
        if not issubclass(arg1_wrapped.__class__, Scalar):
            raise TypeError('First argument to ' + code_str + ' must be a scalar expression, got:\n' + str(arg1_wrapped))
        if not issubclass(arg2_wrapped.__class__, Scalar):
            raise TypeError('Second argument to ' + code_str + ' must be a scalar expression, got:\n' + str(arg2_wrapped))

        # cast constants according to the type of the other input
        arg1_is_constant = type(arg1_wrapped) == _ConstScalar
        arg2_is_constant = type(arg2_wrapped) == _ConstScalar

        if not arg1_is_constant and not arg2_is_constant:
            arg1_expr = arg1_wrapped
            arg2_expr = arg2_wrapped
        elif not arg1_is_constant and arg2_is_constant:
            arg1_expr = arg1_wrapped
            arg2_expr = cast(arg2_wrapped, arg1_wrapped.dtype)
        elif arg1_is_constant and not arg2_is_constant:
            arg1_expr = cast(arg1_wrapped, arg2_wrapped.dtype)
            arg2_expr = arg2_wrapped
        else:
            raise TypeError('Cannot apply binary operator to two constants.')

        t1 = arg1_expr.proto_expr.dtype
        t2 = arg2_expr.proto_expr.dtype
        if not t1 == t2:
            t1_str = lang.DType.Name(t1)
            t2_str = lang.DType.Name(t2)
            raise TypeError('arg1 type (' + t1_str + ') must be the same as arg2 type (' + t2_str + ')')

        if arg1_expr.dtype.proto_dtype not in list(_BinaryMath.code_map[expr_code].keys()):
            raise ValueError(lang.DType.Name(arg1_expr.dtype.proto_dtype) +
                             ' arguments not supported for binary math function ' +
                             lang.ExpressionCode.Name(expr_code))

        super(self.__class__, self).__init__(expr_code, arg1_expr.dtype)

        self.input_exprs = [arg1_expr, arg2_expr]
        super(self.__class__, self)._register()

    @staticmethod
    def from_proto(proto, input_exprs):
        return _BinaryMath(input_exprs[0], input_exprs[1], proto.code)

    def gen_c(self):
        func = _BinaryMath.code_map[self.proto_expr.code][self.dtype.proto_dtype]
        func_str = func(self.input_exprs[0].name, self.input_exprs[1].name)
        return self.dtype.as_cstr() + ' ' + self.name + ' = ' + func_str + ';\n'


def minimum(x, y):
    return _BinaryMath(x, y, lang.MIN)


def maximum(x, y):
    return _BinaryMath(x, y, lang.MAX)


def power(x, y):
    return _BinaryMath(x, y, lang.POW)


def arctan2(x, y):
    return _BinaryMath(x, y, lang.ATAN2)


def logical_and(x, y):
    return _BinaryMath(x, y, lang.AND)


def logical_or(x, y):
    return _BinaryMath(x, y, lang.OR)


class LocalTensor(_TensorExpression, _Readable, _Writable):
    """
    Expression which references a worker-local tensor
    """
    def __init__(self, initial_value):

        if type(initial_value) is not _ConstTensor:
            raise TypeError('Tensors must be initialized by ConstTensors')

        super(self.__class__, self).__init__(lang.TENSOR, initial_value.tensor_type)

        self.input_exprs = [initial_value]

        super(self.__class__, self)._register()

    @staticmethod
    def from_proto(proto, input_exprs):
        return LocalTensor(input_exprs[0])

    def gen_ptr(self):
        tipe = self.dtype.as_cstr()
        name = self.name
        elems = self.size
        return string.Template('${tipe} ${name}[${elems}]').substitute(locals())

    def gen_c(self):
        return self.gen_ptr() + ' = ' + self.input_exprs[0].name + ';\n'


def zeros(shape, dtype):
    """
    Declare a new worker-local tensor with all elements initialized to zero.

    :param shape: the tensor shape
    :param dtype: the tensor data type
    :return: the tensor expression
    """
    np_dtype = DType(dtype).as_numpy()
    init = _ConstTensor(np.zeros(shape, dtype=np_dtype))
    return LocalTensor(init)


def ones(shape, dtype):
    """
    Declare a new worker-local tensor with all elements initialized to one.

    :param shape: the tensor shape
    :param dtype: the tensor data type
    :return: the tensor expression
    """
    np_dtype = DType(dtype).as_numpy()
    init = _ConstTensor(np.ones(shape, dtype=np_dtype))
    return LocalTensor(init)


def _check_index(target_expr, index_expr):
    """
    helper function for making sure that an index is valid
    :param target_expr: the target tensor
    :param index_expr: the index
    :return: the index, wrapped as an expression if necessary
    """

    if issubclass(index_expr.__class__, _Expression):
        index = index_expr
    else:
        index = _ConstScalar(index_expr)

    if index.proto_expr.dtype is lang.UNDEFINED_TYPE:
        raise TypeError('Can only index with a scalar.')

    if type(index) is _ConstScalar:
        if target_expr.size <= index.value() or index.value() < 0:
            raise IndexError('Index out of bounds.')

    return index


class _AssignTensor(_Expression):
    """
    Expression for assigning to tensors
    """
    def __init__(self, tensor_expr, index_expr, value_expr):
        super(self.__class__, self).__init__(lang.ASSIGN_TENSOR)

        if not issubclass(tensor_expr.__class__, _Writable):
            raise TypeError('Can only assign to writable tensors.')

        index = _check_index(tensor_expr, index_expr)

        # try to wrap value as an expression if it's not
        if issubclass(value_expr.__class__, _Expression):
            value = value_expr
        else:
            value = _ConstScalar(value_expr)
            value = cast(value, tensor_expr.dtype)

        # make sure that value is same type as tensor
        t1 = tensor_expr.proto_expr.tensor_type.dtype
        t2 = value.proto_expr.dtype
        if not t1 == t2:
            t1_str = lang.DType.Name(t1)
            t2_str = lang.DType.Name(t2)
            raise TypeError('cannot assign ' + t2_str + ' to ' + t1_str + ' tensor')

        self.input_exprs = [tensor_expr, index, value]
        super(self.__class__, self)._register()

    @staticmethod
    def from_proto(proto, input_exprs):
        return _AssignTensor(input_exprs[0], input_exprs[1], input_exprs[2])

    def gen_c(self):
        return self.input_exprs[0].name + '[' + self.input_exprs[1].name + '] = ' + self.input_exprs[2].name + ';\n'


class _ReadTensor(Scalar):
    """
    Expression for reading from tensors
    """
    def __init__(self, tensor_expr, index_expr):
        if not issubclass(tensor_expr.__class__, _Readable):
            raise TypeError('Can only index a readable tensor.')

        index = _check_index(tensor_expr, index_expr)

        super(self.__class__, self).__init__(lang.READ_TENSOR, tensor_expr.dtype)

        self.input_exprs = [tensor_expr, index]
        super(self.__class__, self)._register()

    @staticmethod
    def from_proto(proto, input_exprs):
        return _ReadTensor(input_exprs[0], input_exprs[1])

    def gen_c(self):
        return self.dtype.as_cstr() + ' ' + self.name + ' = ' + self.input_exprs[0].name + '['+self.input_exprs[1].name+'];\n'


def arange(start, stop=None, step=None):
    """
    Create an iterator to iterate over a range

    :param start: The starting point in the iterator
    :param stop: The stopping point in the iterator
    :param step: The iterator step size
    :return: None

    :Example:

    usage for accumulating a variable to 10::

        accum = variable(0, uint64)
        for i in arange(10):
            accum <<= accum + 1

    """
    if stop is None:
        start_inferred = 0
        stop_inferred = start
    else:
        start_inferred = start
        stop_inferred = stop

    if step is None:
        step_inferred = 1
    else:
        step_inferred = step

    # try to cast all non-expressions as constants
    input_exprs = []
    first_type = None
    for val in [start_inferred, stop_inferred, step_inferred]:
        if issubclass(val.__class__, _Expression):
            input_exprs.append(val)
            if first_type is None:
                first_type = val.dtype
        else:
            input_exprs.append(_ConstScalar(val))

    if first_type is None:
        first_type = _ConstScalar(start).dtype

    # cast all constants as the first dtype
    cast_exprs = []
    for expr in input_exprs:
        if type(expr) is _ConstScalar:
            cast_exprs.append(cast(expr, first_type))
        else:
            cast_exprs.append(expr)

    index = variable(0, first_type)
    return _Range(index, cast_exprs[0], cast_exprs[1], cast_exprs[2])


class _Range(_Expression, six.Iterator):
    """
    A range expression
    """
    def __init__(self, index, start, stop, step):

        self.block_done = False

        first_type = index.dtype
        for expr in [start, stop, step]:
            if expr.dtype != first_type:
                raise TypeError('All input expressions must have the same type.')

        super(self.__class__, self).__init__(lang.RANGE)
        self.input_exprs = [index, start, stop, step]
        super(self.__class__, self)._register()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.block_done:
            self.block_done = True
            return self.input_exprs[0]
        else:
            _EndRange()
            raise StopIteration

    @staticmethod
    def from_proto(proto, input_exprs):
        return _Range(*input_exprs)

    def gen_c(self):
        index_name = self.input_exprs[0].name
        start_name = self.input_exprs[1].name
        stop_name = self.input_exprs[2].name
        step_name = self.input_exprs[3].name

        for_string = 'for(${index_name} = ${start_name}; ' \
                     '((${index_name} < ${stop_name})&&(${step_name}>0)) || ' \
                     '((${index_name} > ${stop_name})&&(${step_name}<0)); ' \
                     '${index_name}+=${step_name}){\n'
        return string.Template(for_string).substitute(locals())


class _EndRange(_Expression):
    """
    The end range expression
    """
    def __init__(self):
        super(self.__class__, self).__init__(lang.ENDRANGE)
        super(self.__class__, self)._register()

    @staticmethod
    def from_proto(proto, input_exprs):
        return _EndRange()

    def gen_c(self):
        return '}\n'


def if_(condition):
    """
    conditional execution, must be used as part of a ``with`` block

    :param condition: The condition under which to execute the body of the with block

    :Example:

    Clip ``input_tensor`` to a maximum value of 1::

        y = variable(0, input_tensor.dtype)
        y = input_tensor[some_index]
        with if_(y > 1):
            y <<= 1
        output_tensor[some_index] = y
    """
    return _If(condition)


class _If(_Expression):
    """
    The if expression
    """
    def __init__(self, condition):
        if not issubclass(condition.__class__, Scalar):
            if isinstance(condition, bool):
                raise TypeError('Attempting to use a constant boolean, %s, with the operator if_ expression. Use the '
                                'python if instead since this can be interpreted at operator '
                                'definition time.' % condition)
            raise TypeError('Condition must be a scalar expression, instead got: ' + str(condition))

        super(self.__class__, self).__init__(lang.IF)
        self.input_exprs = [condition]
        super(self.__class__, self)._register()

    @staticmethod
    def from_proto(proto, input_exprs):
        return _If(input_exprs[0])

    def gen_c(self):
        return 'if('+self.input_exprs[0].name+'){\n'

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        _EndIf()


def elif_(condition):
    """
    else if conditional execution, must be used as part of a ``with`` block and must come directly after
    another if or else if block.

    :param condition: The condition under which to execute the body of the with block

    :Example:

    Clip ``input_tensor`` to a maximum value of 1 and a minimum value of -1::

        y = variable(0, input_tensor.dtype)
        y = input_tensor[some_index]
        with if_(y > 1):
            y <<= 1
        with elif_(y <-1):
            y <<= -1
        output_tensor[some_index] = y

    :param condition: The condition under which to execute the body of the with block
    :return: None
    """
    return _ElseIf(condition)


class _ElseIf(_Expression):
    """
    The elif expression
    """
    def __init__(self, condition):
        if not issubclass(condition.__class__, Scalar):
            raise TypeError('Condition must be a scalar expression')

        super(self.__class__, self).__init__(lang.ELSEIF)
        self.input_exprs = [condition]
        super(self.__class__, self)._register()

    @staticmethod
    def from_proto(proto, input_exprs):
        return _ElseIf(input_exprs[0])

    def gen_c(self):
        return '}\nelse if('+self.input_exprs[0].name+'){\n'

    def __enter__(self):
        ExpressionDAG.remove_endif()

    def __exit__(self, exc_type, exc_val, exc_tb):
        _EndIf()


def else_():
    """
    else conditional execution, must be used as part of a ``with`` block and must come directly after
    another if or else if block.

    :Example:

    Clip ``input_tensor`` to a maximum value of 1 and a minimum value of -1, and zero it
    out if it is within that range::

        y = variable(0, input_tensor.dtype)
        with if_(y > 1):
            y <<= 1
        with elif_(y <-1):
            y <<= -1
        with else_():
            y <<= 0
        output_tensor[some_index] = y

    """
    return _Else()


class _Else(_Expression):
    """
    The else expression
    """
    def __init__(self):
        super(self.__class__, self).__init__(lang.ELSE)
        super(self.__class__, self)._register()

    @staticmethod
    def from_proto(proto, input_exprs):
        return _Else()

    def gen_c(self):
        return '}\nelse{\n'

    def __enter__(self):
        ExpressionDAG.remove_endif()

    def __exit__(self, exc_type, exc_val, exc_tb):
        _EndIf()


class _EndIf(_Expression):
    """
    The endif expression
    """
    def __init__(self):
        super(self.__class__, self).__init__(lang.ENDIF)
        super(self.__class__, self)._register()

    @staticmethod
    def from_proto(proto, input_exprs):
        return _EndIf()

    def gen_c(self):
        return '}\n'