/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.api

import org.diffkt.*

interface DataCollection {
    val size: Int
    operator fun get(index: Int): Pair<FloatTensor, FloatTensor>
}

/** Convenience implementation of DataCollection for a List of feature-label Pairs */
class DataList(val data: List<Pair<FloatTensor, FloatTensor>>) : DataCollection {
    override val size: Int = data.size
    override fun get(index: Int): Pair<FloatTensor, FloatTensor> = data[index]
}

/** Convenience implementation of DataCollection for tensors of features and labels */
class DataTensor(val features: FloatTensor, val labels: FloatTensor) : DataCollection {
    init {
        require(features.shape[0] == labels.shape[0]) {
            "number of features (${features.shape[0]}) does not match number of labels (${labels.shape[0]})" }
    }

    override val size: Int = features.shape[0]

    override fun get(index: Int): Pair<FloatTensor, FloatTensor> {
        return Pair(features[index] as FloatTensor, labels[index] as FloatTensor)
    }
}
