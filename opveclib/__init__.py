# Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
# the specific language governing permissions and limitations under the License.

# @ Data types
from .expression import DType
from .expression import float32, float64
from .expression import int8, int16, int32, int64
from .expression import uint8, uint16, uint32, uint64
from .expression import TensorType

# @ Expressions
from .expression import Scalar, Variable, InputTensor, OutputTensor, PositionTensor, LocalTensor

# @ Tensor functions
from .expression import position_in
from .expression import output, output_like
from .expression import zeros, ones

# @ Scalar functions
# @@ Utility
from .expression import variable, cast
# @@ Unary math
from .expression import arccos, arcsin, arctan, cos, cosh, sin, sinh, tan, tanh
from .expression import exp, log, log10, sqrt
from .expression import ceil, absolute, floor
# @@ Binary math
from .expression import minimum, maximum, power, arctan2, logical_and, logical_or, logical_not

# @ Control flow
from .expression import arange
from .expression import if_, elif_, else_

# @ Operator functions
from .operator import operator, gradient, evaluate, profile, as_tensorflow

# @ Local runtime environment functions
from .local import version, logger, cuda_enabled, cache_directory, clear_op_cache
