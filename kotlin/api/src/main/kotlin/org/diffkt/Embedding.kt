/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/**
 * The backing operation for the embedding layer.
 *
 * Accepts:
 * - @param table Shape(numEmbeddings, embeddingShape)
 * - @param indices Shape(*)
 *
 * @return a [DTensor] of Shape(*, embeddingShape)
 */
fun embedding(
    table: DTensor,
    indices: IntTensor,
    paddingIndex: Int = -1
): DTensor {
    val embeddingShape = table.shape.drop(1)
    return if (indices.shape.isScalar)
            table[indices.data[0]]
        else
            table.gather(indices.data.asList(), 0, paddingIndex).reshape(indices.shape + embeddingShape)
}
