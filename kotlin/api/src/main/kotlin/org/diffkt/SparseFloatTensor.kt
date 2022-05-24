/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.external.SparseOps
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

/**
 * DimData represents a dimension in a SparseFloatTensor.
 *
 * Inner declares the index of an element along this dimension i.e. which index in its row is element i
 * Outer declares the boundaries each component of the dimension i.e which range of elements in inner represent row i
 *
 * See: https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
 */
data class DimData(val inner: IntArray, val outer: IntArray)

interface Sparse

// TODO https://github.com/facebookincubator/diffkt/issues/94: Remove SparseFloatVector and incorporate into SparseFloatTensor cleverly
@SType("S: Shape")
@AllowUnreduced
class SparseFloatVector(
    override val shape: @SType("S") Shape,
    internal val values: FloatArray,
    internal val inner: IntArray
): @SType("S") FloatTensor(), Sparse {
    init {
        require(shape.rank == 1) { "SparseFloatVector must have rank 1" }
    }
    override fun at(pos: Int): Float {
        if(shape.product() <= pos || pos < 0) throw IndexOutOfBoundsException()
        val idx = inner.indexOfFirst { it == pos }
        if (idx == -1) return 0f
        return values[idx]
    }

    override val operations: Operations
        get() = StridedFloatTensorOperations // TODO: Is this right?

    override fun get(indices: IntArray): DTensor {
        if (indices.size != 1) throw IndexOutOfBoundsException()
        return FloatScalar(at(indices[0]))
    }
}

@SType("S: Shape")
@AllowUnreduced
class SparseFloatTensor(
    override val shape: @SType("S") Shape,
    internal val values: FloatArray,
    internal val dims: List<DimData>
): @SType("S") FloatTensor(), Sparse {

    init {
        require(shape.rank > 1) { "SparseFloatTensor must have rank 2 or greater" }
        require(shape.rank == dims.size + 1) {"Provided information for ${dims.size} dimensions, but shape is $shape."}
    }

    override fun get(indices: IntArray): DTensor {
        fun indexingError(indices: IntArray, shape: Shape): String {
            return "Invalid index ${indices.toList()} into SparseTensor with shape $shape"
        }
        require(indices.size == shape.rank) {indexingError(indices, shape)}
        (0 until shape.rank).forEach { require(indices[it] < shape[it] && indices[it] >= 0) { indexingError(indices, shape) } }
        var currElem = 0
        var idxs = dims[currElem].outer.sliceArray(indices[currElem] until indices[currElem] + 2)
        (0 until dims.size).forEach {
            // Use boundaries to slice inner array
            val indicesIntoInner = dims[currElem].inner.sliceArray(idxs[0] until idxs[1])
            currElem += 1
            // Find where the requested value is in the inner array
            val indexOf = indicesIntoInner.indexOf(indices[currElem])
            // Return 0 if it isn't found
            if (indexOf == -1) return FloatScalar(0f)
            // Find the beginning of the next boundary definition
            val nextStart = indexOf + idxs[0]
            // If we've iterated through all the dimensions, return the value
            if (it == dims.size - 1) return FloatScalar(values[nextStart])
            // Otherwise, move the boundaries based on start
            idxs = dims[currElem].outer.sliceArray(nextStart until nextStart + 2)
        }
        return FloatScalar(0f)
    }

    private var invokedElems: List<Pair<IntArray, Float>>? = null

    val nonZeroIndices: List<Pair<IntArray, Float>>
        get() {
            return invokedElems ?: TODO("Conversion not yet written")
        }

    fun toDense(): DTensor {
        return this.normalize()
    }

    internal fun toSparseEigen2D(): SparseEigen2D {
        require(dims.size == 1)
        return SparseEigen2D(shape.dims, values, dims[0].inner, dims[0].outer)
    }

    override fun at(pos: Int): Float {
        require(pos < shape.product || pos < 0) { "Index out of bounds: $pos for shape $shape" }
        return (get(posToIndex(pos)) as FloatScalar).value
    }

    override fun map(f: (Float) -> Float): FloatTensor {
        if (f(0f) != 0f) return (this.toDense() as FloatTensor).map(f)
        val out = SparseFloatTensor(shape, values.map { f(it) }.toFloatArray(), dims)
        if (invokedElems != null) out.invokedElems = invokedElems
        return out
    }

    override fun zip(right: FloatTensor, f: (Float, Float) -> Float): FloatTensor {
        // TODO: Require f(0f, any float) = 0f
        require(f(0f, 1f) == 0f && f(0f, -2.1f) == 0f)
        val newValues = nonZeroIndices.map { f(it.second, right.getAt(it.first)) }
        val out = SparseFloatTensor(shape, newValues.toFloatArray(), dims)
        if (invokedElems != null) out.invokedElems = invokedElems
        return out
    }

    override val operations: Operations
        get() = SparseFloatTensorOperations

    override fun toString(): String {
        return "SparseFloatTensor(values: ${values.toList()}, dims: ${dims})"
    }

    companion object {
        /**
         * Constructor that takes a list of index, value pairs
         */
        operator fun invoke(shape: Shape, elements: List<Pair<IntArray, Float>>): SparseFloatTensor {
            val res = if (shape.rank == 2) SparseOps.convertToCoo(shape, elements)
            else {
                val info = getDimInfo(shape, elements)
                SparseFloatTensor(shape, info.first, info.second)
            }
            res.invokedElems = elements
            return res
        }

        /**
         * Helper function for getDimInfo that merges a list of DimDatas to represent
         * one dimension in the hierarchy
         */
        private fun mergeDimData(data: List<DimData>): DimData {
            val inner = data.fold(IntArray(0)) { acc, e -> acc + e.inner}
            val outer = data.fold(IntArray(1) {0} ) { acc, e ->
                val newE = e.outer.drop(1).map { it + acc.last() }
                acc + newE
            }
            return DimData(inner, outer)
        }

        /**
         * Helper function for [index, value] constructor.
         */
        private fun getDimInfo(shape: Shape, elements: List<Pair<IntArray, Float>>): Pair<FloatArray, List<DimData>> {
            val dims = mutableListOf<DimData>()
            var elems = listOf(elements)
            var dense = true
            var s = shape
            while (s.rank >= 2) {
                val info = elems.map {
                    // How many possible indices there are at this dimension.
                    val firsts = if (dense) (0 until s[0]) else it.map { it.first[0] }.distinct()
                    dense = false
                    // Gather the indices by their first element, but drop the first element to leave the rest
                    val groupedIndices = firsts.map { dim -> it.filter { it.first[0] == dim }.map { Pair(it.first.drop(1).toIntArray(), it.second) } }
                    // Record which distinct values follow each potential start
                    val distinctNext = groupedIndices.map { it.distinctBy { it.first[0] } }
                    // How many distinct values follow each potential start
                    val sizes = distinctNext.map { it.size }
                    // Create outer from sizes
                    val outer = (0 until sizes.size + 1).map { sizes.take(it).sum() }
                    // Create inner by concatenating all the distinct nexts
                    val inner = distinctNext.flatten().map { it.first[0] }
                    val dim1 = DimData(inner.toIntArray(), outer.toIntArray())
                    Pair(dim1, groupedIndices)
                }
                s = s.drop(1)
                // Use the remaining values as elements in the next iteration of the while loop
                elems = info.flatMap{ it.second }
                val allDims = info.map { it.first }
                dims.add(mergeDimData(allDims))
            }
            val values = elems.map{ it.map{ it.second } }.flatten()
            return Pair(values.toFloatArray(), dims)
        }
    }

    // TODO https://github.com/facebookincubator/diffkt/issues/15: A print function
}

/**
 * This class is used in EigenOps.cpp and changes here
 * will need to be reflected there as well.
 */
@AllowUnreduced
internal class SparseEigen2D {
    var shape: IntArray = intArrayOf()
        private set
    var values: FloatArray = floatArrayOf()
        private set
    var inner: IntArray = intArrayOf()
        private set
    var outer: IntArray = intArrayOf()
        private set

    fun toSparseFloatTensor(): SparseFloatTensor {
        return SparseFloatTensor(Shape(shape), values, listOf(DimData(inner, outer)))
    }

    companion object {
        operator fun invoke(shape: IntArray, values: FloatArray, inner: IntArray, outer: IntArray): SparseEigen2D {
            val a = SparseEigen2D()
            a.shape = shape; a.inner = inner; a.outer = outer; a.values = values
            return a
        }
    }
}
