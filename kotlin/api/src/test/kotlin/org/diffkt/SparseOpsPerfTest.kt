/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeExactly
import io.kotest.matchers.ints.shouldBeExactly
import io.kotest.matchers.ints.shouldBeGreaterThanOrEqual
import io.kotest.matchers.ints.shouldBeInRange
import org.diffkt.*
import testutils.shouldBeExactly
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.system.measureTimeMillis

class SparseOpsPerfTest: AnnotationSpec() {
    private val matrixwidthBegin = 1000
    private val matrixwidthEnd = 1024000
    private val bandsize = 101
    private val defaultIter = 10

    // A timer function based on DnnlPerfTest written by @alannnna
    private fun <T> time(
        fn: () -> T) : Float = measureTimeMillis {
        fn()
    }.toFloat()

    // run tests on [fn] with a range of matrix size and runs [iter]+1 iterations for each
    private fun runtests(
        fn: (x : DTensor, y : DTensor) -> DTensor,
        fnname: String,
        iter: Int = defaultIter
    ) {
        var matrixwidth = matrixwidthBegin
        while (matrixwidth <= matrixwidthEnd) {
            (0 until iter + 1).forEach {
                val t1 = bandTensorCSR(Shape(matrixwidth, matrixwidth), bandsize)
                val t2 = bandTensorCSR(Shape(matrixwidth, matrixwidth), bandsize)
                val ms = time{ fn(t1, t2) }
                println("PerfTest: ${fnname} ${it} th run with matrix width ${matrixwidth} band-size ${bandsize} took ${ms / 1000} seconds")
            }
            matrixwidth *= 2
        }
    }

    /**
     * run tests on [fn] with a range of matrix size and runs [iter]+1 iterations for each
     *
     * [fn] takes one COO as input and generate CSR as output
     */
    private fun runtestscooTocsr(
        iter: Int = defaultIter
    ) {
        var matrixwidth = matrixwidthBegin
        while (matrixwidth <= 512000) {
            (0 until iter + 1).forEach {
                val t1 = bandTensorCOO(Shape(matrixwidth, matrixwidth), bandsize)
                val ms = time{ SparseFloatTensor(Shape(matrixwidth, matrixwidth), t1) }
                println("PerfTest: cootocsr ${it} th run with matrix width ${matrixwidth} band-size ${bandsize} took ${ms / 1000} seconds")
            }
            matrixwidth *= 2
        }
    }

    /** Given [shape] and [sparsity], generate a band matrix/tensor with band width as
     * [sparsity] * [shape.dims.last()] */
    private fun bandTensorCSR(shape: Shape, sparsity: Float): SparseFloatTensor {
        return bandTensorCSR(shape, (sparsity*shape.dims.last()).toInt())
    }

    /** Given [shape] and [bandWidth], generate a band matrix/tensor with band width as
     * [bandWidth]
     *
     * when the tensor required is 2D, then a band matrix is generated, see: https://en.wikipedia.org/wiki/Band_matrix
     * when the tensor required is 3D or more, then multiple band matrices are generated.
     * For example, if shape = (x, y, z, w), then, x*y of exactly the same (z, w) band matrices are generated.
     *
     * This function is mainly used in performance testing for sparse computation, such as testing parallelized
     * batched sparse computations, as it can quickly generate large sparse 3D tensors.
     * It can also be used to test the computational correctness for larger matrices.
     * Such as, generating large diagonal matrices for matrix-matrix division correctness testing.
     */
    private fun bandTensorCSR(shape: Shape, bandWidth: Int): SparseFloatTensor {
        shape.rank shouldBeGreaterThanOrEqual 2
        bandWidth shouldBeInRange (1 until shape.dims.last()*2)

        val dims = mutableListOf<DimData>()
        // generate the inners and outers except for the last dim
        var halfBandWidth = floor(bandWidth*0.5).toInt()
        // the number of band matrices.
        val nonEmpty2D = shape.dropLast(2).product
        val nonEmptyRowsIn2D = min(shape.dims[shape.rank - 2], shape.dims.last() + halfBandWidth - (bandWidth + 1) % 2)
        var count = 1
        for (d in 0 until shape.rank - 2) {
            count *= shape[d]
            val nonEmptyRows = if (d == shape.rank - 3) nonEmptyRowsIn2D else shape[d + 1]
            val outer = IntArray(count + 1)
            val inner = IntArray(count * nonEmptyRows)
            var offset = 0
            outer[0] = 0
            for (i in 0 until count) {
                for (j in 0 until nonEmptyRows) inner[offset++] = j
                outer[i+1] = offset
            }
            dims.add(DimData(inner, outer))
        }

        // generate values, and the inner and outer for the last dim
        val start = {r : Int -> max(0, min(r - halfBandWidth + (bandWidth + 1) % 2, shape.dims.last())) }
        val end = {r : Int -> min(shape.dims.last(), r + halfBandWidth + 1) }
        val totalNnz = nonEmpty2D * (0 until shape.dims[shape.rank - 2]).fold(0) {
                acc, r -> acc + end(r) - start(r)
        }
        var values = FloatArray(totalNnz) { 1F }
        var inner = IntArray(totalNnz)
        var outersizeIn2D = (if (shape.rank == 2) shape.dims[shape.rank - 2] else nonEmptyRowsIn2D)
        var outer = IntArray(nonEmpty2D*outersizeIn2D+1)
        var offset = 0
        outer[0] = 0
        for (m in 0 until nonEmpty2D) {
            for (r in 0 until outersizeIn2D) {
                for (c in start(r) until end(r)) {
                    inner[offset] = c
                    offset += 1
                }
                outer[m*outersizeIn2D + r + 1] = offset
            }
        }
        dims.add(DimData(inner, outer))

        return SparseFloatTensor(shape, values, dims)
    }

    /**
     * This function is similar with bandTensorCSR except it generates in a COO format.
     * This is mainly used for performance testing to coo to csr conversion.
     */
    private fun bandTensorCOO(shape: Shape, bandWidth: Int): List<Pair<IntArray, Float>> {
        // Note that since COO and CSR conversion are only supporting 2D,
        // thus the COO generation here are also just support 2D for simplicity
        shape.rank shouldBeExactly  2
        bandWidth shouldBeInRange (1 until shape.dims.last()*2)

        var halfBandWidth = floor(bandWidth*0.5).toInt()
        val nonEmptyRowsIn2D = min(shape.dims[shape.rank - 2], shape.dims.last() + halfBandWidth - (bandWidth + 1) % 2)
        val start = {r : Int -> max(0, min(r - halfBandWidth + (bandWidth + 1) % 2, shape.dims.last())) }
        val end = {r : Int -> min(shape.dims.last(), r + halfBandWidth + 1) }
        var l = mutableListOf<Pair<IntArray, Float>>()
        for (r in 0 until nonEmptyRowsIn2D) {
            for (c in start(r) until end(r)) {
                l.add(Pair(intArrayOf(r, c), 1F))
            }
        }
        return l
    }

    @Test
    fun `test gen bandtensor 2D 5by5 with band width as 3`() {
        val t1 = bandTensorCSR(Shape(5, 5), 1)
        t1 shouldBeExactly tensorOf(
            1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 1.0F).reshape(Shape(5, 5))
    }

    @Test
    fun `test gen bandtensor 2D 5by3 with band width as 3`() {
        val t1 = bandTensorCSR(Shape(5, 3), 3)
        t1 shouldBeExactly tensorOf(
            1.0F, 1.0F, 0.0F,
            1.0F, 1.0F, 1.0F,
            0.0F, 1.0F, 1.0F,
            0.0F, 0.0F, 1.0F,
            0.0F, 0.0F, 0.0F).reshape(Shape(5, 3))
    }

    @Test
    fun `test gen bandtensor 2D 3by5 with band width as 2`() {
        val t1 = bandTensorCSR(Shape(3, 5), 2)
        t1 shouldBeExactly tensorOf(
            1.0F, 1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 1.0F, 0.0F).reshape(Shape(3, 5))
    }

    @Test
    fun `test gen bandtensor 3D`() {
        val t1 = bandTensorCSR(Shape(2, 5, 3), 2)
        t1 shouldBeExactly tensorOf(
            1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 1.0F,
            1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F).reshape(Shape(2, 5, 3))
    }

    @Test
    fun `test gen bandtensor 4D`() {
        val t1 = bandTensorCSR(Shape(2, 2, 4, 5), 3)
        t1 shouldBeExactly tensorOf(
            1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F,
            1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F,
            1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F,
            1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F).reshape(Shape(2, 2, 4, 5))
    }

    @Test
    fun `test gen COO bandtensor 2D 3by5 with band width as 2`() {
        val coo = bandTensorCOO(Shape(3, 5), 2)
        val exp = listOf(Pair(intArrayOf(0, 0), 1f),
            Pair(intArrayOf(0, 1), 1f),
            Pair(intArrayOf(1, 1), 1f),
            Pair(intArrayOf(1, 2), 1f),
            Pair(intArrayOf(2, 2), 1f),
            Pair(intArrayOf(2, 3), 1f))

        (0 until exp.size).forEach {
            for (i in exp[it].first.indices) {
                exp[it].first[i] shouldBeExactly coo[it].first[i]
            }
            exp[it].second shouldBeExactly  coo[it].second
        }
    }

    /** An example of performance testing.
     *
     * To run this as a test, change "@Ignore" to "@Test".
     *
     * To checkout the performance with different number of OpenMP threads,
     * you can (replace [n] with the number of threads you prefer):
     * - update the environment variable "OMP_NUM_THREADS" in Run
     *   configurations with "OMP_NUM_THREADS=n"
     * - or edit the binaryCall function with:
     *    "omp_set_num_threads(n);"
     * - or any better suggestions would be appreciated!!
     */
    @Ignore
    fun `test performance for 3D matmul`() {
        val t1 = bandTensorCSR(Shape(1000, 1000, 1000), 50)
        val t2 = bandTensorCSR(Shape(1000, 1000, 1000), 50)
        val ms = time { t1.matmul(t2) }
        println("Runtime for matmul with 1000by1000by1000 banded tensor with 50 as bandwidth")
        println("with ${System.getenv("OMP_NUM_THREADS")} OpenMP threads: ${ms}ms")
    }

    @Ignore
    fun `test performance for 2D`() {
        runtests({ x, y -> x.matmul(y) }, "matmul")
        runtests({ x, y -> x.times(y) }, "times")
        runtests({ x, y -> x.minus(y) }, "minus")
        runtests({ x, y -> x + y }, "add")
        runtests({ x, _ -> x.transpose() }, "transpose")
        runtestscooTocsr()
    }
}
