/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.api

import org.diffkt.data.Data
import org.diffkt.*
import kotlin.math.min

class SimpleDataIterator(
    val features: FloatTensor,
    val labels: FloatTensor,
    val batchSize: Int = 1,
): Iterable<Data> {
    init {
        require(features.shape.first == labels.shape.first)
    }

    private val n = features.shape.first

    fun withBatchSize(batchSize: Int) = SimpleDataIterator(features, labels, batchSize)

    override fun iterator(): Iterator<Data> = object : Iterator<Data> {
        var loc = 0
        override fun hasNext(): Boolean = loc < n
        override fun next(): Data {
            require(hasNext())

            val start = loc
            val end = min(loc + batchSize, n)
            val f = features.slice(start, end)
            val l = labels.slice(start, end)
            loc = end
            return Data(f, l)
        }
    }
}
