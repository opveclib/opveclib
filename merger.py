from __future__ import print_function
import numpy as np
from opveclib.test.test_operator_dag import split, add, mul, concatenate
from opveclib.operator import _build_op_dag, _merge
from opveclib import language_pb2 as lang

def getOutputIndicies(expDag):
    outs = []
    io_last = -1
    for iExpr, expr in zip(range(len(expDag.expressions._values)), expDag.expressions._values):
        if expr.code == lang.OUTPUT:
            outs.append(iExpr)
            assert(expr.io_index>io_last) # Checks the ordering constraint!
            io_last = expr.io_index
    return outs

def getOutputShape(expDag, iOut):
    return expDag.expressions._values[iOut].tensor_type.shape

nRow = 4
nCol = 5
in0 = np.random.random([nRow, nCol])
row0, row1, row2, row3 = split(in0)
sum0 = add(row0, row1)
sum1 = add(row2, row3)
prod0 = mul(sum0, sum1)
out0 = concatenate(sum0, prod0, sum1)

opDag = _build_op_dag(out0)
#_merge(opDag)

dag = opDag.proto_dag
ins = opDag.inputs
ops = dag.operators._values
outs = dag.dag_outputs._values
refs = dag.references._values

c_ref = refs[4]
first_ref = c_ref.input_refs[0]
print('First inputs op_index = %d.' % first_ref.op_index)
print('First inputs op_output_index = %d.' % first_ref.op_output_index)
print('First inputs dag_input_index = %d.' % first_ref.dag_input_index)

c_op = ops[4]
c_inexp = c_op.expressions._values[first_ref.dag_input_index]
c_inexp.tensor_type.shape
c_outs = getOutputIndicies(c_op)
print('Outputs: ', c_outs)
print('Shape of 0th output', getOutputShape(c_op,c_outs[0]))

# Walk the dag from each output to inputs and compute information about merging of ops.
inputs = [] # This is a list (ab)used as queue
staged = set() # This is a set to hold information about all ops already staged for processing
for output in outs:
    iOutput = output.op_index
    inputs.append(iOutput)
    staged.add(iOutput)
    while len(inputs) > 0:
        i = inputs.pop(0)
        op_out = ops[i]
        ref_out = refs[i]
        #exp_out = op_out.expressions._values[ref_parent.dag_input_index]
        #print('%d',exp_parent.tensor_type.shape[0])
        workgroup_shape_out = op_out.workgroup_shape
        print('index: %d, name: %s.' % (i, op_out.name))
        for input in ref_out.input_refs:
            i = input.op_index
            print('op_index = %d, op_output_index = %d, dag_input_index = %d.' % (i, input.op_output_index, input.dag_input_index))
            expr = op_out.expressions._values[input.dag_input_index]
            print('Shape of this input = ', expr.tensor_type.shape)
            op_in = ops[i]
            in_outs = getOutputIndicies(op_in) # Output indices of the input
            print('The input is %s.' % op_in.name)
            for iIo in in_outs:
                print('\tShape of outputs of input:', getOutputShape(op_in, iIo))
            print('\t\tThe output to this input has the shape:', getOutputShape(op_in, in_outs[input.op_output_index]))


            workgroup_shape_in = op_in.workgroup_shape
            # Only append if not yet processed.
            if i not in staged:
                staged.add(i)
                inputs.append(i)

