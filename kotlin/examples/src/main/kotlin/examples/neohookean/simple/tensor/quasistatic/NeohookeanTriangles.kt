/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.tensor.quasistatic

import org.diffkt.*

data class IndexTriplet(val a: Int, val b: Int, val c: Int)

class NeohookeanTriangles(
    val triangleIndices: List<IndexTriplet>,
    val restShapeInv: DTensor,
    numVertices: Int,
    val mu: Float = 1f,
    val lambda: Float = 1f,
    val restArea: Float = 1f
) {
    val numTriangles get() = triangleIndices.size
    val elementDim get() = 2
    val incidence = makeTriangleIncidence(triangleIndices, numVertices)

    fun energy(x: DTensor): DScalar {
        val shape = incidence.matmul(x)
            .reshape(numTriangles, elementDim, elementDim)
            .transpose(intArrayOf(0, 2, 1))
        val deformationGradient = shape.matmul(restShapeInv)
        return deformationGradientToEnergy(deformationGradient)
    }

    fun makeTriangleIncidence(triangleIndices: List<IndexTriplet>, numVertices: Int): FloatTensor {
        val numTriangles = triangleIndices.size
        val incidenceList = MutableList(numTriangles * elementDim * numVertices) { 0.0f }
        for (i in triangleIndices.indices) {
            val offset = i * elementDim * numVertices
            val triangle = triangleIndices[i]
            incidenceList[offset +               triangle.a] = -1f
            incidenceList[offset +               triangle.b] =  1f
            incidenceList[offset + numVertices + triangle.a] = -1f
            incidenceList[offset + numVertices + triangle.c] =  1f
        }
        return FloatTensor(
            Shape(numTriangles, elementDim, numVertices),
            incidenceList.toFloatArray()
        )
    }

    fun det2d(m: DTensor): DTensor {
        val a = m.view(index = 0, axis = 1).view(index = 0, axis = 1)
        val b = m.view(index = 0, axis = 1).view(index = 1, axis = 1)
        val c = m.view(index = 1, axis = 1).view(index = 0, axis = 1)
        val d = m.view(index = 1, axis = 1).view(index = 1, axis = 1)
        return a * d - b * c
    }

    fun qlog(x: DTensor): DTensor {
        return -1.5f + 2f * x - 0.5f * x.pow(2)
    }

    fun deformationGradientToEnergy(F: DTensor): DScalar {
        val I1 = F.pow(2).sum(1).sum(1)
        val J = det2d(F)
        val qlogJ = qlog(J)
        val energyDensityMu = mu * (0.5f * (I1 - 2f) - qlogJ)
        val energyDensityLambda = lambda * 0.5f * qlogJ.pow(2)
        return restArea * (energyDensityMu + energyDensityLambda).sum()
    }
}
