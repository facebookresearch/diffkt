# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from numpy.linalg import inv
import argparse

import utils

from addsubtimes import addsubtimes
from matmul import matmul
from matdiv import matdiv

def gen(p, nonzerosRatioA, nonzerosRatioB):

    B = utils.genRandomFloatSparseMatrix(p, p, nonzerosRatioA)
    A = utils.genRandomFloatSparseMatrix(p, p, nonzerosRatioB)

    AaddB, AsubB, AtimesB = addsubtimes(A, B)
    return A, B, AaddB, AsubB, AtimesB, matmul(A, B), matdiv(A, B)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='generate random sparse matrices, A, B with given non-zero probability, \
            then perform C =A +/-/times/matmul/matdiv B (A: p by p, B: p by p, C: p by p)')
    parse.add_argument('--p', '-p', type=int, default=3, help='the number of rows/cols for the matrix A/B/C')
    parse.add_argument('--nonzerosRatios', '-z', nargs="+", type=float, default=[0.5, 0.5], help='the ratios of non-zeros for A, B')
    parse.add_argument('--randomseed', '-s', type=int, default=0, help='the seed for numpy to generate random data')
    args = parse.parse_args()

    np.random.seed(args.randomseed)

    nonzerosRatioA, nonzerosRatioB = utils.processListTwo(args.nonzerosRatios, 0.5)
    A, B, AaddB, AsubB, AtimesB, AmatmulB, AmatdivB = gen(args.p, nonzerosRatioA, nonzerosRatioB)

    print("----CPP format----")
    utils.printCSRMatrixCPP(A, 'input A', 'A')
    utils.printCSRMatrixCPP(B, 'input B', 'B')
    utils.printCSRMatrixCPP(AaddB, 'expected add output', 'Add')
    utils.printCSRMatrixCPP(AsubB, 'expected sub output', 'Sub')
    utils.printCSRMatrixCPP(AtimesB, 'expected times output', 'Times')
    utils.printCSRMatrixCPP(AmatmulB, 'expected matmul output', 'Matmul')
    utils.printCSRMatrixCPP(AmatdivB, 'expected matdiv output', 'Matdiv')
    print("----CPP format----")
