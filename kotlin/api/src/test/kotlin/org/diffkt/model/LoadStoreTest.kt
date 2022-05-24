/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.comparables.shouldBeEqualComparingTo
import io.kotest.matchers.should
import org.diffkt.*
import java.nio.ByteBuffer
import java.util.*
import kotlin.random.Random

class LoadStoreTest : AnnotationSpec() {
    @Test
    fun `check random complex model is same with same seed`() {
        val m1 = ComplexModel(testModelRank, Random(shapeSeed), Random(12345))
        val m2 = ComplexModel(testModelRank, Random(shapeSeed), Random(54321))
        m1.isSame(m2) shouldBeEqualComparingTo false
        val m3 = ComplexModel(testModelRank, Random(shapeSeed), Random(12345))
        m1.isSame(m3) shouldBeEqualComparingTo true
    }

    @Test
    fun `check writing and then reading model results in the same thing`() {
        val m1 = ComplexModel(testModelRank, Random(shapeSeed), Random(12345))
        val m2 = ComplexModel(testModelRank, Random(shapeSeed), Random(54321))
        m1.isSame(m2) shouldBeEqualComparingTo false
        val buf = m1.store(ByteBuffer.allocate(1000))
        buf.flip()
        val m3 = m2.load(buf)
        m1.isSame(m3) shouldBeEqualComparingTo true
    }

    companion object {

        const val testModelRank = 15
        const val shapeSeed = 221

        class ComplexModel(override val layers: List<Layer<*>>) : Model<ComplexModel>() {
            override fun hashCode(): Int = combineHash("ComplexModel", layers)
            override fun equals(other: Any?): Boolean = other is ComplexModel && other.layers == layers
            override fun withLayers(newLayers: List<Layer<*>>): ComplexModel {
                return ComplexModel(newLayers)
            }

            fun isSame(other: ComplexModel): Boolean {
                return layers.size == other.layers.size &&
                        layers.zip(other.layers).all {
                            val (a, b) = it
                            a is ComplexTrainableLayer &&
                                    b is ComplexTrainableLayer &&
                                    a.isSame(b)
                        }
            }

            companion object {
                operator fun invoke(
                    n: Int,
                    shapeRandom: Random,
                    valuesRandom: Random
                ): ComplexModel {
                    val q = kotlin.math.sqrt(n.toDouble()).toInt() + 1
                    val layers = (0 until n/q).map { ComplexTrainableLayer(q, shapeRandom, valuesRandom) }
                    return ComplexModel(layers)
                }
            }
        }

        class ComplexTrainableLayer(override val trainables: List<Trainable<*>>) : TrainableLayer<ComplexTrainableLayer> {
            override fun invoke(vararg inputs: DTensor): DTensor {
                TODO("Not yet implemented")
            }

            override fun withTrainables(trainables: List<Trainable<*>>): ComplexTrainableLayer {
                return ComplexTrainableLayer(trainables)
            }

            fun isSame(other: ComplexTrainableLayer): Boolean {
                return this.equals(other)
            }

            override fun hashCode(): Int = combineHash("ComplexTrainableLayer", trainables)
            override fun equals(other: Any?): Boolean = other is ComplexTrainableLayer &&
                    other.trainables == trainables

            companion object {
                private fun makeTrainables(
                    n: Int,
                    shapeRandom: Random,
                    valuesRandom: Random): List<Trainable<*>> {
                    if (n <= 1) {
                        val rank = shapeRandom.nextInt(3)
                        val shape = Shape((0 until rank).map { shapeRandom.nextInt(1, 5) })
                        val tensor = FloatTensor.random(valuesRandom, shape)
                        return listOf(TrainableTensor(tensor))
                    }
                    val nLeft = shapeRandom.nextInt(1, n)
                    val nRight = n - nLeft
                    return listOf(
                        ComplexTrainableLayer(nLeft, shapeRandom, valuesRandom),
                        ComplexTrainableLayer(nRight, shapeRandom, valuesRandom))
                }
                operator fun invoke(
                    n: Int,
                    shapeRandom: Random,
                    valuesRandom: Random): ComplexTrainableLayer {
                    return ComplexTrainableLayer(makeTrainables(n, shapeRandom, valuesRandom))
                }
            }
        }
    }
}
