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

def addsubtimes(A, B):
    AaddB = A + B
    AsubB = A - B
    AtimesB = A.multiply(B)
    return AaddB, AsubB, AtimesB

def gen(p, q, nonzerosRatioA, nonzerosRatioB):

    B = utils.genRandomFloatSparseMatrix(p, q, nonzerosRatioA)
    A = utils.genRandomFloatSparseMatrix(p, q, nonzerosRatioB)

    AaddB, AsubB, AtimesB = addsubtimes(A, B)
    return A, B, AaddB, AsubB, AtimesB

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='generate random sparse matrices, A, B with given non-zero probability, \
            then perform C =A +/-/times B (A: p by q, B: p by q, C: p by q)')
    parse.add_argument('--rowsA', '-p', type=int, default=3, help='the number of rows for the input matrix A/B')
    parse.add_argument('--colsA', '-q', type=int, default=3, help='the number of columns for the input matrix A/B')
    parse.add_argument('--nonzerosRatios', '-z', nargs="+", type=float, default=[0.5, 0.5], help='the ratios of non-zeros for A, B')
    parse.add_argument('--randomseed', '-s', type=int, default=0, help='the seed for numpy to generate random data')
    args = parse.parse_args()

    np.random.seed(args.randomseed)

    nonzerosRatioA, nonzerosRatioB = utils.processListTwo(args.nonzerosRatios, 0.5)
    A, B, AaddB, AsubB, AtimesB = gen(args.rowsA, args.colsA, nonzerosRatioA, nonzerosRatioB)

    print("----CPP format----")
    utils.printCSRMatrixCPP(A, 'input A', 'A')
    utils.printCSRMatrixCPP(B, 'input B', 'B')
    utils.printCSRMatrixCPP(AaddB, 'expected add output', 'Add')
    utils.printCSRMatrixCPP(AsubB, 'expected sub output', 'Sub')
    utils.printCSRMatrixCPP(AtimesB, 'expected times output', 'Times')
    print("----CPP format----")

    print("----Kotlin format----")
    utils.printCOOMatrixKotlin(coo_matrix(A), 'input left', 't1')
    utils.printCOOMatrixKotlin(coo_matrix(B), 'input right', 't2')
    utils.printCOOMatrixKotlin(coo_matrix(AaddB), 'expected add output', 'Add')
    utils.printCOOMatrixKotlin(coo_matrix(AsubB), 'expected sub output', 'Sub')
    utils.printCOOMatrixKotlin(coo_matrix(AtimesB), 'expected times output', 'Times')
    print("----Kotlin format----")
