/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import java.util.*
import shapeTyping.annotations.NoSType

/**
 * A helper class to assign unique integers to each value in a list.
 * You can also think of it like a list with an efficient [indexOf] operation.
 * Uses reference equality, not [Object.equals] to identify the object in the list.
 */
internal class SequentialIntegerAssigner<T: Any> {
    private var nextInteger: Int = 0
    private val map: IdentityHashMap<T, Int> = IdentityHashMap()
    private val keys: MutableList<T> = mutableListOf()

    val values get(): List<T> = keys

    operator fun get(i: Int) = keys[i]

    val size get() = keys.size

    fun add(key: T) {
        val assignedValue = nextInteger
        nextInteger++
        map.put(key, assignedValue)
        keys.add(key)
    }

    fun indexOf(key: Any?): Int {
        return map.get(key) ?: -1
    }

    @NoSType // shapeTyping #111: Generics in same module as calls result in unsubstituted Kotlin types
    inline fun <Q> map(crossinline f: (T) -> Q): List<Q> = keys.map(f)
}
