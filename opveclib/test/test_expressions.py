# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import unittest
from ..expression import *
from ..expression import _Expression, InputTensor, OutputTensor, _ConstScalar, _ConstTensor, Variable, LocalTensor
from ..expression import _AssignVariable, _AssignTensor, _ReadTensor, _UnaryMath, _Cast
from ..expression import _BinaryMath, _to_scalar_index, PositionTensor, _Range, _If, _ElseIf, _Else, _EndIf


def catch_error(f, x, err):
    try:
        f(x)
    except err:
        pass
    else:
        raise AssertionError('Evaluation of f should raise ' + str(err))


class TestTensorExpression(unittest.TestCase):
    def test_dtype(self):
        catch_error(lambda x: DType(x), 'string', TypeError)
        catch_error(lambda x: DType(x), 1.0, TypeError)
        catch_error(lambda x: DType(x), np.float128, TypeError)
        catch_error(lambda x: DType(x), np.ndarray, TypeError)

        catch_error(lambda x: DType(x), 100000, ValueError)

        # numpy types should correspond to defined dtypes
        assert float16 == DType(np.float16)
        assert float32 == DType(np.float32)
        assert float64 == DType(np.float64)
        assert int8 == DType(np.int8)
        assert int16 == DType(np.int16)
        assert int32 == DType(np.int32)
        assert int64 == DType(np.int64)
        assert uint8 == DType(np.uint8)
        assert uint16 == DType(np.uint16)
        assert uint32 == DType(np.uint32)
        assert uint64 == DType(np.uint64)
        assert uint8 == DType(np.zeros(10, dtype=np.uint8).dtype)

        assert float16.as_numpy() == np.float16
        assert float32.as_numpy() == np.float32
        assert float64.as_numpy() == np.float64
        assert int8.as_numpy() == np.int8
        assert int16.as_numpy() == np.int16
        assert int32.as_numpy() == np.int32
        assert int64.as_numpy() == np.int64
        assert uint8.as_numpy() == np.uint8
        assert uint16.as_numpy() == np.uint16
        assert uint32.as_numpy() == np.uint32
        assert uint64.as_numpy() == np.uint64

    def test_tensor_type(self):
        catch_error(lambda x: TensorType(x, float32), 'string', TypeError)
        catch_error(lambda x: TensorType(x, float32), (1, 2, 3.3), TypeError)
        catch_error(lambda x: TensorType(x, float32), np.array([1, 2, 3]), TypeError)
        catch_error(lambda x: TensorType(x, float32), [0, 5], ValueError)

        assert TensorType((1, 2, 3), float32) == TensorType([1, 2, 3], float32)
        assert TensorType((1, 2, 3), float32) == TensorType([1, 2, 3], np.float32)

        assert TensorType(5, float32) == TensorType([5], float32)
        assert TensorType(5, float32) != TensorType([5], float16)

        assert TensorType(5, float32).size == 5
        assert TensorType((5, 10), float32).size == 50
        assert TensorType((5, 10, 100), float32).size == 5000

        assert TensorType.like(np.zeros(10, dtype=np.uint8)) == TensorType(10, uint8)
        assert TensorType.like(np.zeros((10, 10), dtype=np.float64)) == TensorType([10, 10], float64)

    def test_tensor_expression(self):
        ExpressionDAG.clear()
        catch_error(lambda x: _Expression(x), -1, ValueError)

    def test_input(self):
        ExpressionDAG.clear()
        a = np.empty((5, 5), dtype=np.uint64)
        a_type = TensorType.like(a)
        in0 = InputTensor(a_type, 0)
        assert in0.proto_expr.io_index == 0
        assert in0.proto_expr.tensor_type.shape == [5, 5]
        assert in0.proto_expr.tensor_type.dtype == uint64.proto_dtype

        catch_error(lambda x: InputTensor(a_type, x), -1, ValueError)
        catch_error(lambda x: InputTensor(a_type, x), 2 ** 32, ValueError)

        catch_error(lambda x: InputTensor(x, 0), 'string', TypeError)
        catch_error(lambda x: InputTensor(a_type, x), 'string', TypeError)

        ExpressionDAG.clear()
        in0_recovered = InputTensor.from_proto(in0.proto_expr, [])
        assert in0_recovered.proto_expr == in0.proto_expr

    def test_output(self):
        ExpressionDAG.clear()
        a = np.empty((5, 5), dtype=np.uint64)
        a_type = TensorType.like(a)
        in0 = OutputTensor(a_type, 0)
        assert in0.proto_expr.io_index == 0
        assert in0.proto_expr.tensor_type.shape == [5, 5]
        assert in0.proto_expr.tensor_type.dtype == uint64.proto_dtype

        in0[0, 0] = 0

        try:
            in0[0, 0]
        except TypeError:
            pass
        else:
            raise AssertionError('Should not be able to read from an output')

        catch_error(lambda x: OutputTensor(a_type, x), -1, ValueError)
        catch_error(lambda x: OutputTensor(a_type, x), 2 ** 32, ValueError)

        catch_error(lambda x: OutputTensor(x, 0), 'string', TypeError)
        catch_error(lambda x: OutputTensor(a_type, x), 'string', TypeError)

        ExpressionDAG.clear()
        in0_recovered = OutputTensor.from_proto(in0.proto_expr, [])
        assert in0_recovered.proto_expr == in0.proto_expr

    def test_const_scalar(self):
        ExpressionDAG.clear()
        catch_error(lambda x: _ConstScalar(x), 'string', TypeError)
        catch_error(lambda x: _ConstScalar(x), np.arange(1), TypeError)

        t = _ConstScalar(1)
        assert t.proto_expr.code == lang.CONST_SCALAR
        assert t.proto_expr.dtype == lang.INT64
        assert len(t.proto_expr.sint64_data) == 1
        assert t.proto_expr.sint64_data[0] == 1

        t_fp = _ConstScalar.from_proto(t.proto_expr, [])
        assert t.proto_expr == t_fp.proto_expr

        t = _ConstScalar(1.1)
        assert t.proto_expr.code == lang.CONST_SCALAR
        assert t.proto_expr.dtype == lang.FLOAT64
        assert len(t.proto_expr.double_data) == 1
        assert t.proto_expr.double_data[0] == 1.1

        t_fp = _ConstScalar.from_proto(t.proto_expr, [])
        assert t.proto_expr == t_fp.proto_expr

        t.proto_expr.dtype = lang.UNDEFINED_TYPE
        catch_error(lambda x: t.value(), None, ValueError)
        catch_error(lambda x: _ConstScalar.from_proto(t.proto_expr, []), None, ValueError)

    def test_const_tensor(self):
        ExpressionDAG.clear()
        catch_error(lambda x: _ConstTensor(x), 'string', TypeError)

        assert _ConstTensor([1, 2, 3]).proto_expr == _ConstTensor((1, 2, 3)).proto_expr
        assert _ConstTensor(np.arange(1, 4, dtype=np.int64)).proto_expr == _ConstTensor((1, 2, 3)).proto_expr
        assert _ConstTensor(np.arange(1, 4, dtype=np.int32)).proto_expr != _ConstTensor((1, 2, 3)).proto_expr

        a = _ConstTensor(np.random.random(10).astype(np.float32))
        a_fp = _ConstTensor.from_proto(a.proto_expr, [])
        assert a.proto_expr == a_fp.proto_expr

        a = _ConstTensor(np.random.randint(low=-2**63, high=2**63-1, size=10).astype(np.int64))
        a_fp = _ConstTensor.from_proto(a.proto_expr, [])
        assert a.proto_expr == a_fp.proto_expr

    def test_position(self):
        ExpressionDAG.clear()
        pos = PositionTensor((5, 5))
        ExpressionDAG.clear()
        pos2 = PositionTensor.from_proto(pos.proto_expr, [])

        assert pos.proto_expr == pos2.proto_expr

    def test_scalar(self):
        ExpressionDAG.clear()
        catch_error(lambda x: variable(x, float32), 'string', TypeError)

        a = variable(1.0, float32)
        assert a.proto_expr.dtype == float32.proto_dtype
        assert len(a.input_exprs) == 1

        # check scalar assignment operator
        a <<= 5
        assignment = ExpressionDAG.exprs[-1]
        scalar_dag_index = ExpressionDAG.expr_index(a)
        assign_to_dag_index = ExpressionDAG.expr_index(assignment.input_exprs[0])
        assert scalar_dag_index == assign_to_dag_index

        ExpressionDAG.clear()
        b = variable(2.0, float32)
        assert b.proto_expr == Variable.from_proto(b.proto_expr, b.input_exprs).proto_expr

    def test_assign_scalar(self):
        ExpressionDAG.clear()
        catch_error(lambda x: _AssignVariable(variable(0, float32), x), 'string', TypeError)
        catch_error(lambda x: _AssignVariable(variable(0, float32), x), variable(0, float64), TypeError)
        catch_error(lambda x: _AssignVariable(x, variable(0, float32)), 'string', TypeError)

        a = variable(1.0, float32)
        b = variable(2.0, float32)
        c = _AssignVariable(a, b)
        assert c.input_exprs[0].proto_expr.dtype == float32.proto_dtype
        assert len(c.input_exprs) == 2

        a = variable(1.0, float32)
        c = _AssignVariable(a, 2)
        assert c.input_exprs[1].proto_expr.dtype == float32.proto_dtype
        assert len(c.input_exprs) == 2

        assert c.proto_expr == _AssignVariable.from_proto(c.proto_expr, c.input_exprs).proto_expr

    def test_cast(self):
        ExpressionDAG.clear()
        catch_error(lambda x: cast(x, float32), 'string', TypeError)
        catch_error(lambda x: cast(x, float32), 1.0, TypeError)
        catch_error(lambda x: cast(x, float32), PositionTensor((5, 5)), TypeError)

        a = variable(1.0, float32)
        b = cast(a, float64)
        assert b.proto_expr.dtype == float64.proto_dtype
        assert len(b.input_exprs) == 1

        assert b.proto_expr == _Cast.from_proto(b.proto_expr, b.input_exprs).proto_expr

    def test_tensor(self):
        ExpressionDAG.clear()
        catch_error(lambda x: LocalTensor(x), 'string', TypeError)

        a = zeros((5, 5), int8)
        assert a.proto_expr.tensor_type.dtype == int8.proto_dtype
        assert len(a.input_exprs) == 1
        b = a.input_exprs[0]
        assert len(b.proto_expr.sint32_data) == 25
        for elem in b.proto_expr.sint32_data:
            assert elem == 0

        a = ones((5, 5), int8)
        assert a.proto_expr.tensor_type.dtype == int8.proto_dtype
        assert len(a.input_exprs) == 1
        b = a.input_exprs[0]
        assert len(b.proto_expr.sint32_data) == 25
        for elem in b.proto_expr.sint32_data:
            assert elem == 1

        assert a.proto_expr == LocalTensor.from_proto(a.proto_expr, a.input_exprs).proto_expr

    def test_assign_tensor(self):
        ExpressionDAG.clear()

        a = zeros(5, dtype=float32)

        # cannot assign to input
        catch_error(lambda x: _AssignTensor(x, 0, 1.0), InputTensor(TensorType.like(a), 0), TypeError)

        # cannot wrap a string as a proper index
        catch_error(lambda x: _AssignTensor(a, x, 1.0), 'string', TypeError)

        # cannot use a tensor as an index
        catch_error(lambda x: _AssignTensor(a, x, 1.0), zeros(1, uint8), TypeError)

        # constant index must be within bounds
        catch_error(lambda x: _AssignTensor(a, x, 1.0), 100, IndexError)
        catch_error(lambda x: _AssignTensor(a, x, 1.0), -1, IndexError)

        # cannot wrap a constant other than int or float
        catch_error(lambda x: _AssignTensor(a, [0], x), 'string', TypeError)

        # cannot assign a differently typed scalar to a tensor
        catch_error(lambda x: _AssignTensor(a, [0], x), variable(1.0, float64), TypeError)

        b = _AssignTensor(a, 1, 1)
        assert b.input_exprs[0] is a
        assert b.input_exprs[1].proto_expr == _ConstScalar(1).proto_expr
        assert b.input_exprs[2].proto_expr == cast(_ConstScalar(1), a.dtype).proto_expr

        n = variable(5, uint8)
        v = variable(1.1, float32)
        b = _AssignTensor(a, n, v)
        assert b.input_exprs[1] is n
        assert b.input_exprs[2] is v

        assert b.proto_expr == _AssignTensor.from_proto(b.proto_expr, b.input_exprs).proto_expr

    def test_index(self):
        ExpressionDAG.clear()
        a = zeros((5, 5), float32)

        catch_error(lambda x: _ReadTensor(a, x), -1, IndexError)
        catch_error(lambda x: _ReadTensor(a, x), 25, IndexError)

        b = OutputTensor(TensorType.like(a), 0)
        catch_error(lambda x: _ReadTensor(b, x), 5, TypeError)

        c = _ReadTensor(a, 0)
        assert c.input_exprs[0] is a
        assert c.input_exprs[1].proto_expr == _ConstScalar(0).proto_expr

        d = variable(0, uint8)
        e = _ReadTensor(a, d)
        assert e.input_exprs[1] is d

        assert e.proto_expr == _ReadTensor.from_proto(e.proto_expr, e.input_exprs).proto_expr

    def test_to_scalar_index(self):
        ExpressionDAG.clear()
        target_shape = (5, 5)

        catch_error(lambda x: _to_scalar_index(target_shape, x), [1], IndexError)
        _to_scalar_index(target_shape, [0, 0])
        catch_error(lambda x: _to_scalar_index(target_shape, x), [0, 0, 0], IndexError)

        catch_error(lambda x: _to_scalar_index(target_shape, x), zeros(1, float32), IndexError)
        _to_scalar_index(target_shape, zeros(2, float32))
        catch_error(lambda x: _to_scalar_index(target_shape, x), zeros(3, float32), IndexError)
        catch_error(lambda x: _to_scalar_index(target_shape, x), zeros((2, 1), float32), IndexError)

        catch_error(lambda x: _to_scalar_index(target_shape, x), [variable(0, uint32)], IndexError)
        _to_scalar_index(target_shape, [variable(0, uint32), 0])
        catch_error(lambda x: _to_scalar_index(target_shape, x), [variable(0, uint32), 0, 0], IndexError)

        catch_error(lambda x: _to_scalar_index(target_shape, x), [variable(0, uint32), 5], IndexError)
        catch_error(lambda x: _to_scalar_index(target_shape, x), [0, 5], IndexError)

        catch_error(lambda x: _to_scalar_index(target_shape, x), _EndIf, TypeError)

    def test_unary_math(self):
        ExpressionDAG.clear()
        a = variable(1.1, float32)
        b = variable(1.1, float64)
        catch_error(lambda x: _UnaryMath(a, x), lang.POSITION, ValueError)
        catch_error(lambda x: _UnaryMath(x, lang.EXP), variable(1.1, float16), ValueError)

        def assert_equivalent(x, y):
            assert x.proto_expr == y.proto_expr
            assert x.input_exprs[0] is y.input_exprs[0]
            assert x.dtype == y.dtype

        # make sure all exposed unary math functions are producing correct expressions for floats and doubles
        def assert_all(x):
            assert_equivalent(arccos(x), _UnaryMath(x, lang.ACOS))
            assert_equivalent(arcsin(x), _UnaryMath(x, lang.ASIN))
            assert_equivalent(arctan(x), _UnaryMath(x, lang.ATAN))
            assert_equivalent(cos(x), _UnaryMath(x, lang.COS))
            assert_equivalent(cosh(x), _UnaryMath(x, lang.COSH))
            assert_equivalent(sin(x), _UnaryMath(x, lang.SIN))
            assert_equivalent(sinh(x), _UnaryMath(x, lang.SINH))
            assert_equivalent(tan(x), _UnaryMath(x, lang.TAN))
            assert_equivalent(tanh(x), _UnaryMath(x, lang.TANH))
            assert_equivalent(exp(x), _UnaryMath(x, lang.EXP))
            assert_equivalent(log(x), _UnaryMath(x, lang.LOG))
            assert_equivalent(log10(x), _UnaryMath(x, lang.LOG10))
            assert_equivalent(sqrt(x), _UnaryMath(x, lang.SQRT))
            assert_equivalent(ceil(x), _UnaryMath(x, lang.CEIL))
            assert_equivalent(floor(x), _UnaryMath(x, lang.FLOOR))
            assert_equivalent(absolute(x), _UnaryMath(x, lang.ABS))
            assert_equivalent(logical_not(x), _UnaryMath(x, lang.NOT))
            assert_equivalent(-x, _UnaryMath(x, lang.NEGATE))

        assert_all(a)
        assert_all(b)

        def assert_int_ok(x):
            assert_equivalent(absolute(x), _UnaryMath(x, lang.ABS))
            assert_equivalent(logical_not(x), _UnaryMath(x, lang.NOT))
            assert_equivalent(-x, _UnaryMath(x, lang.NEGATE))

        assert_int_ok(variable(1, int8))
        assert_int_ok(variable(1, int16))
        assert_int_ok(variable(1, int32))
        assert_int_ok(variable(1, int64))

        c = absolute(a)
        assert c.proto_expr == _UnaryMath.from_proto(c.proto_expr, c.input_exprs).proto_expr

    def test_binary_math(self):
        ExpressionDAG.clear()

        a = variable(1.1, float32)
        b = variable(1.2, float32)

        catch_error(lambda x: _BinaryMath(a, x, lang.ADD), variable(0, float64), TypeError)
        catch_error(lambda x: _BinaryMath(1, x, lang.ADD), 2, TypeError)
        catch_error(lambda x: _BinaryMath(a, b, x), lang.POSITION, ValueError)

        c = _BinaryMath(a, b, lang.ADD)
        assert c.input_exprs[0] is a
        assert c.input_exprs[1] is b
        assert c.dtype == float32

        d = _BinaryMath(a, 1, lang.ADD)
        assert d.dtype == float32

        d = _BinaryMath(1, a, lang.ADD)
        assert d.dtype == float32

        # make sure all magic methods invoke the right expression
        def assert_eqivalent(x, y):
            assert x.proto_expr == y.proto_expr
            assert x.input_exprs[0].proto_expr == y.input_exprs[0].proto_expr
            assert x.input_exprs[1].proto_expr == y.input_exprs[1].proto_expr

        assert_eqivalent(a + 1, _BinaryMath(a, 1, lang.ADD))
        assert_eqivalent(1 + a, _BinaryMath(1, a, lang.ADD))
        assert_eqivalent(a - 1, _BinaryMath(a, 1, lang.SUBTRACT))
        assert_eqivalent(1 - a, _BinaryMath(1, a, lang.SUBTRACT))
        assert_eqivalent(a * 1, _BinaryMath(a, 1, lang.MULTIPLY))
        assert_eqivalent(1 * a, _BinaryMath(1, a, lang.MULTIPLY))
        assert_eqivalent(a / 1, _BinaryMath(a, 1, lang.DIVIDE))
        assert_eqivalent(1 / a, _BinaryMath(1, a, lang.DIVIDE))
        assert_eqivalent(a % 1, _BinaryMath(a, 1, lang.MODULO))
        assert_eqivalent(1 % a, _BinaryMath(1, a, lang.MODULO))
        assert_eqivalent(a == 1, _BinaryMath(a, 1, lang.EQUAL))
        assert_eqivalent(1 == a, _BinaryMath(a, 1, lang.EQUAL))
        assert_eqivalent(a != 1, _BinaryMath(a, 1, lang.NOTEQUAL))
        assert_eqivalent(1 != a, _BinaryMath(a, 1, lang.NOTEQUAL))
        assert_eqivalent(a < 1, _BinaryMath(a, 1, lang.LESS))
        assert_eqivalent(1 < a, _BinaryMath(a, 1, lang.GREATER))
        assert_eqivalent(a <= 1, _BinaryMath(a, 1, lang.LESS_EQ))
        assert_eqivalent(1 <= a, _BinaryMath(a, 1, lang.GREATER_EQ))
        assert_eqivalent(a > 1, _BinaryMath(a, 1, lang.GREATER))
        assert_eqivalent(1 > a, _BinaryMath(a, 1, lang.LESS))
        assert_eqivalent(a >= 1, _BinaryMath(a, 1, lang.GREATER_EQ))
        assert_eqivalent(1 >= a, _BinaryMath(a, 1, lang.LESS_EQ))

        assert c.proto_expr == _BinaryMath.from_proto(c.proto_expr, c.input_exprs).proto_expr

        a = variable(1.0, float32)
        b = variable(1.0, float32)
        catch_error(lambda x: _BinaryMath(a, b, x), lang.POSITION, ValueError)
        catch_error(lambda x: _BinaryMath(variable(1.0, x), b, lang.POW), lang.FLOAT16, TypeError)
        catch_error(lambda x: _BinaryMath(variable(1.0, x), variable(1.0, x), lang.POW), lang.FLOAT16, TypeError)
        catch_error(lambda x: _BinaryMath(2, 2, lang.POW), lang.FLOAT16, TypeError)

        def assert_equivalent(x, y):
            assert x.proto_expr == y.proto_expr
            assert x.input_exprs[0] is y.input_exprs[0]
            assert x.input_exprs[1] is y.input_exprs[1]
            assert x.dtype == y.dtype

        for t in [float32, float64]:
            a1 = variable(1.0, t)
            a2 = variable(1.0, t)
            assert_equivalent(power(a1, a2), _BinaryMath(a1, a2, lang.POW))
            assert_equivalent(arctan2(a1, a2), _BinaryMath(a1, a2, lang.ATAN2))

        for t in supported_types:
            a1 = variable(1.0, t)
            a2 = variable(1.0, t)
            assert_equivalent(minimum(a1, a2), _BinaryMath(a1, a2, lang.MIN))
            assert_equivalent(maximum(a1, a2), _BinaryMath(a1, a2, lang.MAX))
            assert_equivalent(logical_or(a1, a2), _BinaryMath(a1, a2, lang.OR))
            assert_equivalent(logical_and(a1, a2), _BinaryMath(a1, a2, lang.AND))

        c = minimum(a, b)
        assert c.proto_expr == _BinaryMath.from_proto(c.proto_expr, c.input_exprs).proto_expr

    def test_arange(self):
        ExpressionDAG.clear()

        def f(x):
            for i in arange(x, variable(10, uint16), 1):
                pass
        f(variable(0, uint16))
        catch_error(f,  variable(0, float32), TypeError)

        a = arange(10)
        assert a.proto_expr == _Range.from_proto(a.proto_expr, a.input_exprs).proto_expr

    def test_if(self):
        ExpressionDAG.clear()
        catch_error(lambda x: _If(x), zeros((5, 5), float32), TypeError)
        catch_error(lambda x: _ElseIf(x), zeros((5, 5), float32), TypeError)
        a = variable(1.0, float32)
        b = variable(0.0, float32)
        _If(a < 5)
        b <<= 2
        _ElseIf(a < 4)
        b <<= 1
        _Else()
        b <<= 0
        _EndIf()

    def test_native_if(self):
        ExpressionDAG.clear()
        a = variable(0.0, float32)
        try:
            if a == 0:
                pass
        except SyntaxError:
            pass
        else:
            raise AssertionError

        try:
            if a:
                pass
        except SyntaxError:
            pass
        else:
            raise AssertionError

        try:
            max(a, 1)
        except SyntaxError:
            pass
        else:
            raise AssertionError

        try:
            min(a, 1)
        except SyntaxError:
            pass
        else:
            raise AssertionError
