/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.linreg

import examples.api.Learner
import examples.api.SimpleDataIterator
import org.diffkt.*
import org.diffkt.model.*
import kotlin.random.Random

class LinearRegression(val l: AffineTransform): Model<LinearRegression>() {
    constructor(m: DScalar, b: DScalar) : this(AffineTransform(TrainableTensor(m), TrainableTensor(b)))
    constructor(random: Random) : this(FloatScalar(random.nextFloat()), FloatScalar(random.nextFloat()))

    override val layers: List<Layer<*>> = listOf(l)

    override fun withLayers(newLayers: List<Layer<*>>): LinearRegression {
        require(newLayers.size == 1)
        val newLayer = newLayers[0] as AffineTransform
        return LinearRegression(newLayer)
    }

    override fun hashCode(): Int = combineHash("LinearRegression", l)
    override fun equals(other: Any?): Boolean = other is LinearRegression &&
            other.l == l
}

const val BATCH_SIZE = 100

fun main() {
    val random = Random(1234567)
    val trueWeight = FloatScalar(random.nextFloat())
    val trueBias = FloatScalar(random.nextFloat())
    val linReg = LinearRegression(random)
    val features = FloatTensor(Shape(BATCH_SIZE)) { random.nextFloat() }
    val labels = (features * trueWeight + trueBias) as FloatTensor
    val optimizer = FixedLearningRateOptimizer<LinearRegression>(0.5F / BATCH_SIZE)
    val dataIterator = SimpleDataIterator(features, labels, BATCH_SIZE)
    fun lossFun(predictions: DTensor, labels: DTensor): DScalar {
        val diff = predictions - labels
        return (diff * diff).sum()
    }
    val learner = Learner(
        batchedData = dataIterator,
        lossFunc = ::lossFun,
        optimizer = optimizer,
        useJit = true
    )
    /*val trainedLinreg = */ learner.train(linReg, 80000, printProgress = false)
    learner.dumpTimes()
}
