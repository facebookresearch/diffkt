/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*

abstract class BatchNormTrainingBase<T: BatchNormTrainingBase<T>>(
    protected val numFeatures: Int,
    protected val momentum: Float = 0.1F,
    protected val scaleShift: TrainableTensor,
): TrainableLayerSingleInput<T>, LayerWithInferenceMode {
    init {
        require(momentum in 0F..1F)
    }

    /**
     * The computed running mean and variance.
     */
    abstract val stats: Pair<DTensor, DTensor>

    /**
     * Freeze the batch norm transform that was used during training,
     * returning an affine transform to be used for inference.
     */
    override val inferenceMode: AffineTransform get() {
        // We use the running stats computed during training
        // rather than processing the training set again to compute new stats as is done
        // in https://arxiv.org/abs/1502.03167 Algorithm 2
        val (mean, variance) = stats
        return freezeBatchNorm(scaleShift.tensor, mean, variance)
    }

    override fun hashCode(): Int = combineHash("BatchNormTrainingBase", numFeatures, momentum, scaleShift)
    override fun equals(other: Any?): Boolean = other is BatchNormTrainingBase<*> &&
            other.numFeatures == numFeatures &&
            other.momentum == momentum &&
            other.scaleShift == scaleShift
}

/**
 * A training version of batch normalization provided for compatibility with existing code.
 */
class BatchNorm2d(
    numFeatures: Int,
    momentum: Float = 0.1F
) : BatchNormTraining(numFeatures, momentum)

/**
 * A trainable Batch Normalization transform, as described in https://arxiv.org/abs/1502.03167 .
 * When training is complete, use its @see [inferenceMode] property to get the computed affine transform.
 * This version maintains an exponential moving average of the sum of the samples, sum of the squared
 * samples, and sample count which are used to estimate the population mean and variance.
 *
 * Epsilon is hardcoded to 1.e-5f.
 */
open class BatchNormTraining private constructor(
    numFeatures: Int,
    momentum: Float = 0.1F,
    scaleShift: TrainableTensor,
    private var runningN: Float,
    private var runningSum: DTensor,
    private var runningSumOfSquares: DTensor,
) : BatchNormTrainingBase<BatchNormTraining>(numFeatures, momentum, scaleShift) {

    constructor(
        numFeatures: Int,
        momentum: Float = 0.1F) : this(
            numFeatures = numFeatures,
            momentum = momentum,
            scaleShift = TrainableTensor(FloatTensor.ones(Shape(1, numFeatures)).concat(FloatTensor.zeros(Shape(1, numFeatures)))),
            runningN = 0F,
            runningSum = FloatTensor.zeros(Shape(numFeatures)),
            runningSumOfSquares = FloatTensor.zeros(Shape(numFeatures)))

    override fun invoke(input: DTensor): DTensor {
        val (output, newStats) = batchNormTrainV2(
            input, scaleShift.tensor, runningN, runningSum, runningSumOfSquares, momentum)
        val (newRunningN, newRunningSum, newRunningSumOfSquares) = newStats

        // TODO https://github.com/facebookincubator/diffkt/issues/172: Since this class is not pure,
        // it is not threadsafe.  Synchronize or refactor the API.
        runningN = newRunningN
        runningSum = newRunningSum
        runningSumOfSquares = newRunningSumOfSquares
        return output
    }

    /**
     * The computed running mean and variance.
     */
    override val stats: Pair<DTensor, DTensor> get() {
        val n = runningN
        val mean = runningSum / n
        val variance = runningSumOfSquares / n - mean.pow(2)
        return Pair(mean, variance)
    }

    override val trainables: List<Trainable<*>>
        get() = listOf(scaleShift)

    override fun withTrainables(trainables: List<Trainable<*>>): BatchNormTraining {
        require(trainables.size == 1)
        val newScaleShift = trainables[0]
        require(newScaleShift is TrainableTensor)
        return BatchNormTraining(numFeatures, momentum, newScaleShift, runningN, runningSum, runningSumOfSquares)
    }
}

/**
 * A trainable Batch Normalization transform, as described in https://arxiv.org/abs/1502.03167 .
 * When training is complete, use its @see [inferenceMode] property to get the computed affine transform.
 * This version is provided to imitate the behavior in V1, the previous implementation, in that
 * it calculates a running mean and running variance rather than gathering the raw input to compute
 * the mean and variance.  It applies Bessel's correction
 * (https://en.wikipedia.org/wiki/Bessel%27s_correction) to the sample variance to get an estimate of the
 * population variance for each batch, and uses an exponential moving average of those values as an
 * estimate the population variance when @see [inferenceMode] is applied.
 *
 * Epsilon is hardcoded to 1.e-5f
 */
class BatchNormTrainingV1 private constructor(
    numFeatures: Int,
    momentum: Float = 0.1F,
    scaleShift: TrainableTensor,
    private var runningMean: DTensor,
    private var runningVariance: DTensor,
) : BatchNormTrainingBase<BatchNormTrainingV1>(numFeatures, momentum, scaleShift) {
    init {
        require(momentum in 0F..1F)
    }

    override fun invoke(input: DTensor): DTensor {
        val (output, newRuningMean, newRunningVariance) = batchNormTrainV1(
            input, scaleShift.tensor, runningMean, runningVariance, momentum)

        // TODO https://github.com/facebookincubator/diffkt/issues/172: Since this class is not pure,
        // it is not threadsafe.  Synchronize or refactor the API.
        runningMean = newRuningMean
        runningVariance = newRunningVariance
        return output
    }

    /**
     * The computed running mean and variance.
     */
    override val stats: Pair<DTensor, DTensor> get() = Pair(runningMean, runningVariance)

    override val trainables: List<Trainable<*>>
        get() = listOf(scaleShift)

    override fun withTrainables(trainables: List<Trainable<*>>): BatchNormTrainingV1 {
        require(trainables.size == 1)
        val newScaleShift = trainables[0]
        require(newScaleShift is TrainableTensor)
        return BatchNormTrainingV1(numFeatures, momentum, newScaleShift, runningMean, runningVariance)
    }

    companion object {
        operator fun invoke(
            numFeatures: Int,
            momentum: Float = 0.1F,
        ): BatchNormTrainingV1 {
            val scale = FloatTensor.ones(Shape(1, numFeatures))
            val shift = FloatTensor.zeros(Shape(1, numFeatures))
            val scaleShift = TrainableTensor(scale.concat(shift))
            val runningMean = FloatTensor.zeros(Shape(numFeatures))
            val runningVariance = FloatTensor.ones(Shape(numFeatures))
            return BatchNormTrainingV1(numFeatures, momentum, scaleShift, runningMean, runningVariance)
        }
    }
}

fun freezeBatchNorm(scaleShift: DTensor, mean: DTensor, variance: DTensor): AffineTransform {
    // Freeze historical mean and variance values during inference.
    val frozenMean = mean.basePrimal()
    val frozenVariance = variance.basePrimal()

    // In computing the standard deviation, we add an epsilon "to improve stability".
    // See "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
    // https://arxiv.org/abs/1502.03167 Algorithm 2 (page 4).
    // This prevents us from dividing by zero when computing m below.
    val stddev = sqrt(frozenVariance + BATCHNORM_EPSILON)
    val scale = scaleShift[0]
    val shift = scaleShift[1]

    // https://arxiv.org/abs/1502.03167 Algorithm 2 Step 11
    val m = scale / stddev
    val b = shift - m * frozenMean
    return AffineTransform(TrainableTensor(m), TrainableTensor(b))
}

/**
 * The batchNorm op for training, V1 compatibility version
 *
 * @param input:  an NHWC tensor
 * @param scaleShift:  the combined scale and shift tensor, with shape (2, C)
 * @return Triple of:
 *   - output, with shape NHWC
 *   - mean over input, with shape C
 *   - sample variance over input, with shape C
 */
fun batchNormTrainV1(
    input: DTensor,
    scaleShift: DTensor,
    runningMean: DTensor,
    runningVariance: DTensor,
    momentum: Float
): Triple<DTensor, DTensor, DTensor> {
    val r = batchNorm(input, scaleShift)
    val n = r.n.toFloat()
    val batchNormResultvariance = r.variance * (n / (n - 1))
    val newRunningMean = runningMean.momentumUpdated(r.mean, momentum)
    val newRunningVariance = runningVariance.momentumUpdated(batchNormResultvariance, momentum)
    return Triple(r.result, newRunningMean, newRunningVariance)
}

/**
 * The batchNorm op for training
 *
 * @param input:  an NHWC tensor
 * @param scaleShift:  the combined scale and shift tensor, with shape (2, C)
 * @return Pair of:
 *   - output, with shape NHWC
 *   - Triple of:
 *       - New running N
 *       - New running sum
 *       - New running sum of squares
 */
fun batchNormTrainV2(
    input: DTensor,
    scaleShift: DTensor,
    runningN: Float,
    runningSum: DTensor,
    runningSumOfSquares: DTensor,
    momentum: Float
): Pair<DTensor, Triple<Float, DTensor, DTensor>> {
    val r = batchNorm(input, scaleShift)
    val newRunningN = runningN.momentumUpdated(r.n, momentum)
    val newRunningSum = runningSum.momentumUpdated(r.sum, momentum)
    val newRunningSumOfSquares = runningSumOfSquares.momentumUpdated(r.sumOfSquares, momentum)
    return Pair(r.result, Triple(newRunningN, newRunningSum, newRunningSumOfSquares))
}
