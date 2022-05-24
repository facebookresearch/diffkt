/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

// ==========
// Transcendentals
// ==========

// Trig
fun sin(x: DScalar): DScalar {
    return x.operations.sin(x) as DScalar
}

fun sin(x: DTensor): DTensor {
    return x.operations.sin(x)
}

fun cos(x: DScalar): DScalar {
    return x.operations.cos(x) as DScalar
}

fun cos(x: DTensor): DTensor {
    return x.operations.cos(x)
}

fun tan(x: DScalar): DScalar {
    return x.operations.tan(x) as DScalar
}

fun tan(x: DTensor): DTensor {
    return x.operations.tan(x)
}

// Other transcendentals
fun atan(x: DScalar): DScalar {
    return x.operations.atan(x) as DScalar
}

fun atan(x: DTensor): DTensor {
    return x.operations.atan(x)
}

fun exp(x: DScalar): DScalar {
    return x.operations.exp(x) as DScalar
}

fun exp(x: DTensor): DTensor {
    return x.operations.exp(x)
}

fun ln(x: DScalar): DScalar {
    return x.operations.ln(x) as DScalar
}

fun ln(x: DTensor): DTensor {
    return x.operations.ln(x)
}

fun lgamma(x: DScalar): DScalar {
    return x.operations.lgamma(x) as DScalar
}

fun lgamma(x: DTensor): DTensor {
    return x.operations.lgamma(x)
}

fun digamma(x: DScalar): DScalar {
    return x.operations.digamma(x) as DScalar
}

fun digamma(x: DTensor): DTensor {
    return x.operations.digamma(x)
}

fun polygamma(n: Int, x: DScalar): DScalar {
    return x.operations.polygamma(n, x) as DScalar
}

fun polygamma(n: Int, x: DTensor): DTensor {
    return x.operations.polygamma(n, x)
}

fun sqrt(x: DScalar): DScalar {
    return x.operations.sqrt(x) as DScalar
}

fun sqrt(x: DTensor): DTensor {
    return x.operations.sqrt(x)
}

fun tanh(x: DScalar): DScalar {
    return x.operations.tanh(x) as DScalar
}

fun tanh(x: DTensor): DTensor {
    return x.operations.tanh(x)
}
