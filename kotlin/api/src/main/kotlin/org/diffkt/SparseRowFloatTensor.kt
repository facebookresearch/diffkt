/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/**
 * A sparse row float tensor, which is a [FloatTensor] of rank 2 which has sparse rows (e.g. has rows that are empty/zero).
 * This data format is particularly useful for embedding table gradients.
 *
 * Example:
 *  For this 4x3 tensor:
 *      7 7 7 7
 *      0 0 0 0
 *      8 8 8 8
 *
 *  The sparse row data will be:
 *      0 7 7 7 7
 *      2 8 8 8 8
 *      (shape: [3, 4])
 *  Note that the rows must be in order by index, so this is not valid (behavior is undefined):
 *      2 8 8 8 8
 *      0 7 7 7 7
 *      (shape: [3, 4])
 */
class SparseRowFloatTensor internal constructor(
    override val shape: Shape,
    internal val data: FloatArray
): FloatTensor() {
    init {
        assert(shape.rank == 2)
    }

    override val operations: Operations get() = SparseRowFloatTensorOperations

    // --- Data Access/Indexing ---
    private val dataWidth = shape[1] + 1
    // Map from row number to position of the marker in the data
    private val rowMap = run {
        val populatedRowsSize = data.size / dataWidth
        (0 until populatedRowsSize).associate {
            val rowIndex = dataWidth * it
            Pair(data[rowIndex].toInt(), rowIndex)
        }
    }

    override fun at(pos: Int): Float {
        // convert to shape indices naively
        val columnSize = shape[1]
        val row = pos / columnSize
        val col = pos % columnSize
        return rowMap[row]?.let {
            data[it + col + 1]
        } ?: 0f
    }

    override fun all(p: (Float) -> Boolean): Boolean {
        if (rowMap.size < shape[1]) {
            if (!p(0f)) return false
        }
        for (i in data.indices) {
            if (i % dataWidth != 0) {
                if (!p(data[i])) return false
            }
        }
        return true
    }

    override fun map(f: (Float) -> Float): FloatTensor {
        // check f(0)
        val zeroMapped = f(0f)
        if (zeroMapped != 0f) {
            val cdata = FloatArray(size)
            val columnSize = shape[1]
            for (row in 0 until shape[0]) {
                val offset = row * columnSize
                rowMap[row]?.let {
                    for (i in 0 until columnSize) {
                        cdata[offset + i] = f(data[it + i + 1])
                    }
                } ?: run {
                    for (i in 0 until columnSize) {
                        cdata[offset + i] = zeroMapped
                    }
                }
            }
            return FloatTensor(shape, cdata)
        }

        val resData = FloatArray(data.size)
        for (index in resData.indices) {
            resData[index] = if (index % dataWidth == 0) data[index] else f(data[index])
        }
        return SparseRowFloatTensor(shape, resData)
    }

    /*
    leftIdentity:       f(0, x) = x
    rightIdentity:      f(x, 0) = x
    leftAnnihilating:   f(0, x) = 0
    rightAnnihilating:  f(x, 0) = 0
     */
    internal fun qualifiedZip(
        right: FloatTensor,
        leftAnnihilating: Boolean,
        rightAnnihilating: Boolean,
        f: (Float, Float) -> Float,
    ): FloatTensor {

        if (right !is SparseRowFloatTensor || f(0f, 0f) != 0f) {
            return super.zip(right, f)
        }

        val left = this
        var leftIndex = 0; var rightIndex = 0
        val leftDataSize = left.data.size; val rightDataSize = right.data.size

        // compute number of rows for resData
        var rowCount = 0
        while (leftIndex < leftDataSize && rightIndex < rightDataSize) {
            val leftRowNumber = left.data[leftIndex]
            val rightRowNumber = right.data[rightIndex]
            when {
                leftRowNumber == rightRowNumber -> {
                    leftIndex+=dataWidth; rightIndex+=dataWidth; rowCount++
                }
                leftRowNumber < rightRowNumber -> {
                    leftIndex+=dataWidth
                    if (!rightAnnihilating) rowCount++
                }
                else -> {
                    rightIndex+=dataWidth
                    if (!leftAnnihilating) rowCount++
                }
            }
        }
        if (!leftAnnihilating) rowCount += (rightDataSize - rightIndex)/dataWidth
        if (!rightAnnihilating) rowCount += (leftDataSize - leftIndex)/dataWidth
        var resData = FloatArray(rowCount * dataWidth)

        // Merge left and right
        var resIndex = 0; leftIndex = 0; rightIndex = 0;
        while (leftIndex < leftDataSize && rightIndex < rightDataSize) {
            val leftRowNumber = left.data[leftIndex]
            val rightRowNumber = right.data[rightIndex]
            var resNonZero = false
            when {
                leftRowNumber == rightRowNumber -> {
                    resData[resIndex] = leftRowNumber
                    for (i in 1 until dataWidth) {
                        val res = f(left.data[leftIndex + i], right.data[rightIndex + i])
                        if (res != 0f) resNonZero = true
                        resData[resIndex + i] = res
                    }
                    leftIndex+=dataWidth; rightIndex+=dataWidth;
                    if (resNonZero) {
                        resIndex+=dataWidth
                    }
                }
                leftRowNumber < rightRowNumber -> {
                    if (!rightAnnihilating) {
                        resData[resIndex] = leftRowNumber
                        for (i in 1 until dataWidth) {
                            val res = f(left.data[leftIndex + i], 0f)
                            if (res != 0f) resNonZero = true
                            resData[resIndex + i] = res
                        }
                        if (resNonZero) {
                            resIndex+=dataWidth
                        }
                    }
                    leftIndex+=dataWidth;
                }
                else -> {
                    if (!leftAnnihilating) {
                        resData[resIndex] = rightRowNumber
                        for (i in 1 until dataWidth) {
                            val res = f(0f, right.data[rightIndex + i])
                            if (res != 0f) resNonZero = true
                            resData[resIndex + i] = res
                        }
                        if (resNonZero) {
                            resIndex+=dataWidth
                        }
                    }
                    rightIndex+=dataWidth;
                }
            }
        }
        if (!rightAnnihilating) {
            while (leftIndex < leftDataSize) {
                var resNonZero = false
                resData[resIndex] = left.data[leftIndex]
                for (i in 1 until dataWidth) {
                    val res = f(left.data[leftIndex + i], 0f)
                    if (res != 0f) resNonZero = true
                    resData[resIndex + i] = res
                }
                leftIndex+=dataWidth;
                if (resNonZero) {
                    resIndex+=dataWidth
                }
            }
        }
        if (!leftAnnihilating) {
            while (rightIndex < rightDataSize) {
                var resNonZero = false
                resData[resIndex] = right.data[rightIndex]
                for (i in 1 until dataWidth) {
                    val res = f(0f, right.data[rightIndex + i])
                    if (res != 0f) resNonZero = true
                    resData[resIndex + i] = res
                }
                rightIndex+=dataWidth;
                if (resNonZero) {
                    resIndex+=dataWidth
                }
            }
        }

        if (resIndex < resData.size) {
            resData = resData.sliceArray(0 until resIndex)
        }

        return if (resData.size == shape[0] * dataWidth)
            StridedFloatTensor(shape, 1, intArrayOf(dataWidth, 1), resData)
        else SparseRowFloatTensor(shape, resData)
    }

    override fun zip(right: FloatTensor, f: (Float, Float) -> Float): FloatTensor {
        return qualifiedZip(
            right,
            leftAnnihilating = false, rightAnnihilating = false,
            f
        )
    }

}
