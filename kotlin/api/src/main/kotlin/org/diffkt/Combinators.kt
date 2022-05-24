/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

object Combinators {
    /**
     * Reduces x along axes by applying function f to reduce the elements along the axes. If keepDims is true,
     * the original rank of x is retained, otherwise the reduced axes are removed in the output tensor.
     *
     * If no dimension is provided, the tensor is reduced on all dimensions, returning a tensor with shape
     * Shape(), a scalar tensor.
     *
     * Example
     * >>> val t1 = FloatTensor(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f))
     * >>> t1.reduce({a: Float, b: Float -> a * b}, listOf(1))
     * FloatTensor(Shape(2), floatArrayOf(4f, 6f))
     */
    fun reduce(
        x: FloatTensor,
        f: (Float, Float) -> Float,
        axes: IntArray = x.allAxes,
        keepDims: Boolean = false
    ): FloatTensor {
        fun cell(data: FloatArray, csize: Int, ix: (Int) -> Int): Float {
            if (data.isEmpty())
                throw IllegalArgumentException("Empty selection cannot be reduced")
            if (data.size == 1 && csize == 1)
                return data[ix(0)]

            var acc = data[ix(0)]
            for (i in 1 until csize) {
                acc = f(acc, data[ix(i)])
            }
            return acc
        }
        if (x is SparseFloatTensor) return reduceSparse(x, f, axes, keepDims)
        return reduceImpl(x, { data, csize, ix -> cell(data, csize, ix) }, axes, keepDims)
    }

    internal fun reduceImpl(
        x: FloatTensor,
        f: (FloatArray, Int, (Int) -> Int) -> Float,
        cellAxes: IntArray,
        keepDims: Boolean = false
    ): FloatTensor {
        if (cellAxes.isEmpty()) return x
        val info = partitionForReduce(x, cellAxes)
        val s = x.asStrided()
        val a = FloatArray(info.frameSize) { i ->
            val base = StridedUtils.strided(i, info.frameShape, s.strides)
            fun ix(j: Int) = base + StridedUtils.strided(j, info.cellShape, s.strides) + s.offset
            f(s.data, info.cellSize) { ix(it) }
        }
        val rsh = (if (keepDims) info.frameShape else Shape(info.frameAxes.map { info.frameShape[it] }))
        val rst = (if (keepDims) info.frameStrides else info.frameAxes.map { info.frameStrides[it] }.toIntArray())
        return if (rsh.isScalar) FloatScalar(a[0]) else StridedFloatTensor(rsh, offset = 0, rst, a)
    }

    internal fun reduceSparse(x: SparseFloatTensor,
                              f: (Float, Float) -> Float,
                              axes: IntArray = x.allAxes,
                              keepDims: Boolean = false): FloatTensor {
        // TODO: a check that f(0, any float) = the float
        require(f(0f, 1f) == 1f && f(0f, -2f) == -2f)
        // TODO: don't require keep dims to be true
        require(keepDims == true) { "reduceSparse requires keepDims = true for now"}
        // Some logic is here for accepting keepDims = false, but some is missing, namely reducing
        // to a rank < 2
        val resShape =
            if (keepDims)
                Shape(x.shape.dims.mapIndexed { ix, it -> if (ix in axes) 1 else it })
            else
                Shape(x.shape.dims.filterIndexed { ix, _ -> ix !in axes })
        val indices = x.nonZeroIndices
        val grouped = indices.groupBy { it.first.filterIndexed { index, _ -> !axes.contains(index)} }
        val newIndices = grouped.map { Pair(it.key, it.value.map { it.second }.reduce(f)) }
        val out = if (keepDims) newIndices.map {
            val indexPart = it.first.toMutableList()
            axes.forEach { indexPart.add(it, 0) }
            Pair(indexPart.toIntArray(), it.second)
        }
        else newIndices.map { Pair(it.first.toIntArray(), it.second) }
        return SparseFloatTensor(resShape, out)
    }

    /**
     * given cell axes, return frame axes plus cell and frame shapes/strides/data sizes
     * result of a reduce will have shape selected from our shape by frame axes,
     * with contents of each cell reduced to a scalar
     */
    private fun partitionForReduce(x: FloatTensor, cellAxes: IntArray): PartitionInfo {
        fun subshape(axs: IntArray): List<Int> {
            val ss = MutableList(x.rank) { 1 }
            axs.forEach { ss[it] = x.shape[it] }
            return ss
        }

        val cell = cellAxes.sorted().toIntArray()
        if (cell.isNotEmpty()) {
            if (cell[0] < 0)
                throw IllegalAccessException("reduce: invalid axis ${cellAxes[0]}")
            if (cell.last() >= x.rank)
                throw IllegalAccessException("reduce: invalid axis ${cellAxes[0]} for shape ${x.shape}")
        }

        val frame = x.shape.indices.filter { !cell.contains(it) }.toIntArray()
        val cshape = Shape(subshape(cell))
        val cstrides = StridedUtils.contigStrides(cshape)
        val csize = StridedUtils.dataSize(cshape, cstrides)
        val fshape = Shape(subshape(frame))
        val fstrides = StridedUtils.contigStrides(fshape)
        val fsize = StridedUtils.dataSize(fshape, fstrides)

        return PartitionInfo(frame, fshape, fstrides, fsize, cshape, cstrides, csize)
    }

    private data class PartitionInfo(
        val frameAxes: IntArray,
        val frameShape: Shape,
        val frameStrides: IntArray,
        val frameSize: Int,
        val cellShape: Shape,
        val cellStrides: IntArray,
        val cellSize: Int
    )
}
