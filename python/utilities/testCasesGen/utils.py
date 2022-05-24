# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

def genRandomFloatSparseMatrix(rows, cols, nonzerosRatio):
    assert nonzerosRatio >= 0 and nonzerosRatio <= 1, 'nonzeros ratio should be in range [0, 1]'
    assert rows > 0 and cols > 0, 'numbers of rows and cols should be both larger than 0'

    vals=np.random.rand(rows, cols)
    mask=np.random.choice(2, size=(rows, cols), p=[1-nonzerosRatio, nonzerosRatio])

    return csr_matrix(vals * mask)

def printLongList(preline, l, sufline, elementsSuffix='', endstr=',', elementsPerLine=5):
    print(preline)
    for x in range(len(l)):
        if (x == len(l) - 1):
            print(str(l[x]) + elementsSuffix)
        else:
            print(str(l[x]) + elementsSuffix, end = endstr)
            if (x+1) % elementsPerLine == 0:
                print()
    print(sufline)

def printCOO3DMatrixKotlin(cooms, objname, suffix):
    if len(cooms) < 1:
        return
    #check on shapes
    shape = [len(cooms), cooms[0].shape[0], cooms[0].shape[1]]
    for i in range(shape[0]):
        if cooms[i].shape[0] != shape[1] or cooms[i].shape[1] != shape[2]:
            print("Shape is inconsistent in printing COO 3D matrix kotlin format for {}".format(objname))
            return

    print('//{}:'.format(objname))
    print('val {} = SparseFloatTensor(Shape({}, {}, {}), listOf('.format(suffix, shape[0], shape[1], shape[2]))
    for i in range(shape[0]):
        for x in range(cooms[i].nnz):
            print('Pair(intArrayOf({}, {}, {}), {}f)'.format(i, cooms[i].row[x], cooms[i].col[x], cooms[i].data[x]), end = '')
            if i != shape[0]-1 or x < cooms[i].nnz - 1:
                print(',')
    print('))')

def printCOOMatrixKotlin(coom, objname, suffix):
    print('//{}:'.format(objname))
    print('val {} = SparseFloatTensor(Shape({}, {}), listOf('.format(suffix, coom.shape[0], coom.shape[1]))
    for x in range(coom.nnz):
        print('Pair(intArrayOf({}, {}), {}f)'.format(coom.row[x], coom.col[x], coom.data[x]), end = '')
        if x < coom.nnz - 1:
            print(',')
    print('))')

def printCSRMatrixKotlin(csrm, objname, suffix):
    print('//{}:'.format(objname))
    print('val shape{} = Shape({}, {})'.format(suffix, csrm.shape[0], csrm.shape[1]));
    printLongList('val values' + suffix + ' = floatArrayOf(', list(csrm.data), ')', 'f');
    printLongList('val inner' + suffix + ' = intArrayOf(', list(csrm.indices), ')');
    printLongList('val outer' + suffix + ' = intArrayOf(', list(csrm.indptr), ')');

def printCSRMatrixCPP(csrm, objname, suffix):
    print('//{}:'.format(objname))
    print('DimensionType rows' + suffix + ' = ' + str(csrm.shape[0]) + ', cols' + suffix + ' = ' + str(csrm.shape[1]) + ';');
    printLongList('std::vector<DataType> values' + suffix + ' = {', list(csrm.data), '};');
    printLongList('std::vector<DimensionType> inner' + suffix + ' = {', list(csrm.indices), '};');
    printLongList('std::vector<OrdinalType> outer' + suffix + ' = {', list(csrm.indptr), '};');


def printCSRMatrix(csrm, objname, suffix):
    printCSRMatrixCPP(csrm, objname, suffix)

def permuteCOOMatrix(coom):
    newOrder = np.random.permutation(coom.nnz)
    newCOOValues = []
    newCOORows = []
    newCOOCols = []
    for i in range(coom.nnz):
        newCOOValues.append(coom.data[newOrder[i]])
        newCOORows.append(coom.row[newOrder[i]])
        newCOOCols.append(coom.col[newOrder[i]])
    return coo_matrix((newCOOValues, (newCOORows, newCOOCols)), shape=coom.shape)

def orderPerservingCOOToCSR(coom):
    csrm = csr_matrix(coom)
    newCSRValues = csrm.data.copy()
    newCSRIndptr = csrm.indptr.copy()
    newCSRIndices = csrm.indices.copy()
    indptr = csrm.indptr.copy()
    for i in range(coom.nnz):
        r = coom.row[i]
        newCSRValues[indptr[r]] = coom.data[i]
        newCSRIndices[indptr[r]] = coom.col[i]
        indptr[r] += 1
    return csr_matrix((newCSRValues, newCSRIndices, newCSRIndptr), shape=coom.shape)

def printCOOMatrixCPP(coom, objname, suffix):
    print('//{}:'.format(objname))
    print('Array<DimensionType> shape{}(2);'.format(suffix));
    print('shape{}[0] = {}, shape{}[1] = {};'.format(suffix, coom.shape[0], suffix, coom.shape[1]))
    print('Array<DimensionType> rowsIndex' + suffix + ' {', end = '');
    for x in range(coom.nnz):
        print('{}'.format(coom.row[x]), end = '')
        if x < coom.nnz - 1:
            print(', ', end='')
    print('};');
    print('Array<DimensionType> colsIndex' + suffix + ' {', end = '');
    for x in range(coom.nnz):
        print('{}'.format(coom.col[x]), end = '')
        if x < coom.nnz - 1:
            print(', ', end='')
    print('};');
    print('Array<DataType> values' + suffix + ' {', end = '');
    for x in range(coom.nnz):
        print('{}'.format(coom.data[x]), end = '')
        if x < coom.nnz - 1:
            print(', ', end='')
    print('};');

# Use for extracting two values [a, b] from a list of values [l]
# [v] is the default value for [a] and [b].
# It works in a way:
# If [l] is empty: [a=v, b=v]
# If [l] has only one element, [x]: [a=x, b=x]
# If [l] has only two or more element [x, y,...] : [a=x, b=y]
#
# Parameter:
# [l] : a list of values
# [v] : the default value
def processListTwo(l, v=0.5):
    a, b = processList(l, 1, v)
    return a[0], b[0]

# Use for extracting two lists of length [s] from a list of values [l]
# [v] is the default value for the list [As] and [Bs].
# It works in a way:
# If [l] has only one element, [x]: [As=[x]*s, Bs=[x]*s]
# If [l] has only two elements, [x, y]: [As=[x]*s, Bs=[y]*s]
# If [l] has only 2*[s] elements, [x_1, .., x_s, y_1, .., y_s] : [As=[x_1, .., x_s], Bs=[y_1, .., y_s]]
#
# Parameter:
# [l] : a list of values
# [s] : the size for the resulting list
# [v] : the default value
# [name] : the name for the list for printing out error info
def processList(l, s, v=0.2, name="non-zero ratio"):
    As = [v] * s
    Bs = [v] * s
    if len(l) == 1:
        As = l * s
        Bs = l * s
    elif len(l) == 2:
        As = [l[0]] * s
        Bs = [l[1]] * s
    elif len(l) == 2*s:
        As = l[0:s]
        Bs = l[s:]
    else:
        print("Given length of {} doesn't fit, so default value ({}) is used for all".format(name, v))

    return As, Bs
