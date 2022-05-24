/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.random.RandomKey
import kotlin.reflect.KClass
import kotlin.reflect.full.allSuperclasses
import kotlin.reflect.full.isSuperclassOf

/**
 * An interface that is capable of wrapping individual values.  To be used by an
 * implementation of [Differentiable] to wrap the individual values.
 */
abstract class Wrapper {
    fun <T> wrap(value: T): T {
        @Suppress("UNCHECKED_CAST")
        return wrapRaw((value ?: return value)::class, value) as T
    }

    abstract fun wrapDTensor(value: DTensor): DTensor

    open fun wrapRandomKey(value: RandomKey): RandomKey = value

    private fun wrapRaw(type: KClass<*>, value: Any?): Any? {
        if (value == null)
            return null
        if (value is Wrappable<*>)
            return value.wrap(this)

        if (type.java.isArray) {
            // Support wrapping arrays.
            val elementType = type.java.componentType.kotlin
            val newArray = java.lang.reflect.Array.newInstance(type.java.componentType, java.lang.reflect.Array.getLength(value))
            for (i in 0 until java.lang.reflect.Array.getLength(value)) {
                java.lang.reflect.Array.set(newArray, i, wrapRaw(elementType, java.lang.reflect.Array.get(value, i)))
            }
            return newArray
        }

        val gWrapper = getWrapper(type, value)
        if (gWrapper == null)
            return value // no wrapper - return without wrapping.
        return gWrapper(value, this)
    }

    companion object {
        private val wrapMap: HashMap<KClass<*>, (Any, Wrapper) -> Any> = HashMap()

        fun getWrapper(type: KClass<*>, value: Any): ((Any, Wrapper) -> Any)? {
            // return any registered wrapper that both
            // 1. Wraps the value's actual type or one of its supertypes, and
            // 2. Wraps type `type` or one of its subtypes
            val actualType: KClass<*> = value::class

            // assert that actualType is a subtype of type
            assert(type.isSuperclassOf(actualType))

            if (wrapMap.containsKey(actualType))
                return wrapMap[actualType]

            // TODO https://github.com/facebookincubator/diffkt/issues/92: should this use the most specific wrapping function?
            // TODO https://github.com/facebookincubator/diffkt/issues/92: should this use "type" to help select the wrapper?
            for (sup in actualType.allSuperclasses) {
                if (wrapMap.containsKey(sup))
                    return wrapMap[sup]
            }

            return null
        }

        /**
         * Register a wrapper so the given type can be used as the input or output of
         * a function that can be differentiated.
         */
        fun <T : Any> register(clazz: KClass<T>, wrapper: (T, Wrapper) -> T) {
            @kotlin.Suppress("UNCHECKED_CAST")
            wrapMap[clazz] = wrapper as (Any, Wrapper) -> Any
        }
        fun registerRaw(clazz: KClass<*>, wrapper: (Any, Wrapper) -> Any) {
            wrapMap[clazz] = wrapper
        }

        init {
            register(List::class) { value: List<*>, wrapper: Wrapper -> value.map { wrapper.wrap(it) } }
            register(Pair::class) { value: Pair<*,*>, wrapper: Wrapper -> Pair(wrapper.wrap(value.first), wrapper.wrap(value.second)) }
        }
    }
}
