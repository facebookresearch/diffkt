/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import org.diffkt.*
import org.diffkt.random.RandomKey
import java.util.ArrayList
import kotlin.reflect.KClass

internal fun <TData : Any> reusedNodes(t: TData): Set<Traceable> {
    val roots = ArrayList<Traceable>()
    val myWrapper = object : Wrapper() {
        override fun wrapDTensor(value: DTensor): DTensor {
            return when (value) {
                is TracingTensor -> {
                    roots.add(value)
                    value
                }
                else -> value
            }
        }

        override fun wrapRandomKey(value: RandomKey): RandomKey {
            return when (value) {
                is TracingRandomKey -> {
                    roots.add(value)
                    value
                }
                else -> value
            }
        }
    }
    myWrapper.wrap(t)
    val counts = useCounts(roots)
    val dups = counts.filter { it.value > 1 }.map { it.key }
    return HashSet(dups)
}
