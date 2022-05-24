/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import java.util.concurrent.atomic.AtomicInteger

/**
 * We identify values produced for forward or reverse differentiation with a DerivativeID.
 * We assign each derivative a unique DerivativeID with its own unique "sequence" number.
 * Each instance of DerivativeID represents a separate set of perturbations,
 * the product of any two of which is zero.
 *
 * In nested derivatives,
 * higher sequence numbers would always be used for the more inner derivative,
 * and appear higher in the tree of Duals (or Reverses) representing a number.
 * A sequence number of zero is used at the leaves where the value was produced
 * without reference to any derivative operation; values for these are just
 * wrappers around the raw data (either [FloatScalar] or [FloatTensor]).
 */
abstract class DerivativeID(
    val sequence:Int = nextSequence.getAndAdd(1)
) {
    override fun toString(): String = "e$sequence"

    companion object {
        /**
         * The next sequence number to assign.
         */
        private var nextSequence: AtomicInteger = AtomicInteger(1)
    }

    val isNone = sequence == 0
}

object NoDerivativeID : DerivativeID(sequence = 0) {
}
