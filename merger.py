from __future__ import print_function
import numpy as np
from opveclib.test.test_operator_dag import split, add, mul, concatenate
from opveclib.operator import _build_op_dag, _merge

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



# Walk the dag from outputs to inputs.
q = [] # This is a list (ab)used as queue
p = set() # This is a set to hold information about all ops already processed
for output in outs:
    iOutput = output.op_index
    q.append(iOutput)
    p.add(iOutput)
    while len(q) > 0:
        i = q.pop(0)
        op_parent = ops[i]
        ref_parent = refs[i]
        #exp_parent = op_parent.expressions._values[ref_parent.dag_input_index]
        #print('%d',exp_parent.tensor_type.shape[0])
        shape_parent = op_parent.workgroup_shape # get the workgroup shape of the parent op
        print('index: %d, name: %s.' % (i, op_parent.name))
        for input in ref_parent.input_refs:
            i = input.op_index
            op_child = ops[i]
            shape_child = op_child.workgroup_shape # workgroup shape of the child op

            # Only append if not yet processed.
            if i not in p:
                p.add(i)
                q.append(i)

