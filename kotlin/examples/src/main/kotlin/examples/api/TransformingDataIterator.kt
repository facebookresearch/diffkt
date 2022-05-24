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

class TransformingDataIterator(
    val data: DataCollection,
    val individualTransform: (DTensor) -> DTensor = { it },
    val batchTransform: (DTensor) -> DTensor = { it },
    val batchSize: Int = 1,
): Iterable<Data> {
    constructor(
        data: List<Pair<FloatTensor, FloatTensor>>,
        individualTransform: (DTensor) -> DTensor = { it },
        batchTransform: (DTensor) -> DTensor = { it },
        batchSize: Int = 1,
    ) : this(DataList(data), individualTransform, batchTransform, batchSize)

    constructor(
        features: FloatTensor,
        labels: FloatTensor,
        individualTransform: (DTensor) -> DTensor = { it },
        batchTransform: (DTensor) -> DTensor = { it },
        batchSize: Int = 1,
    ) : this(DataTensor(features, labels), individualTransform, batchTransform, batchSize)

    fun withBatchSize(batchSize: Int) = TransformingDataIterator(data, individualTransform, batchTransform, batchSize)

    val featureShape = data[0].first.shape
    val labelShape = data[0].second.shape

    override fun iterator(): Iterator<Data> = object : Iterator<Data> {
        var loc = 0
        override fun hasNext(): Boolean = loc < data.size
        override fun next(): Data {
            require(hasNext())
            val featureTensors = mutableListOf<DTensor>()
            val labelTensors = mutableListOf<DTensor>()

            val start = loc
            val end = min(loc + batchSize, data.size)
            assert(end <= data.size)
            loc = end
            for (i in start until end) {
                val (feature, label) = data[i]
                val transformedFeature = individualTransform(feature)
                featureTensors.add(transformedFeature)
                labelTensors.add(label)
            }

            val features = concat(featureTensors.map() { it.unsqueeze(0) })
            assert(features.shape == Shape(batchSize) + featureShape)
            val labels = concat(labelTensors.map() { it.unsqueeze(0) })
            assert(labels.shape == Shape(batchSize) + labelShape)

            val transformedFeatures = batchTransform(features)
            return Data(transformedFeatures as FloatTensor, labels as FloatTensor)
        }
    }
}
