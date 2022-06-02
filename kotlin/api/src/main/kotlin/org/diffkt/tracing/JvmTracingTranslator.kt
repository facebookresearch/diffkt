/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import net.bytebuddy.ByteBuddy
import net.bytebuddy.asm.AsmVisitorWrapper
import net.bytebuddy.description.field.FieldDescription
import net.bytebuddy.description.field.FieldList
import net.bytebuddy.description.method.MethodDescription
import net.bytebuddy.description.method.MethodList
import net.bytebuddy.description.type.TypeDescription
import net.bytebuddy.dynamic.scaffold.InstrumentedType
import net.bytebuddy.implementation.Implementation
import net.bytebuddy.implementation.bytecode.*
import net.bytebuddy.implementation.bytecode.collection.ArrayAccess
import net.bytebuddy.implementation.bytecode.constant.FloatConstant
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant
import net.bytebuddy.implementation.bytecode.member.MethodInvocation
import net.bytebuddy.implementation.bytecode.member.MethodReturn
import net.bytebuddy.implementation.bytecode.member.MethodVariableAccess
import net.bytebuddy.jar.asm.*
import net.bytebuddy.matcher.ElementMatchers
import net.bytebuddy.pool.TypePool
import org.diffkt.*
import org.diffkt.external.External
import org.diffkt.random.RandomKey
import org.diffkt.sigmoidElem
import java.lang.Math.max
import java.util.*
import kotlin.reflect.KFunction1
import kotlin.reflect.KFunction2
import kotlin.reflect.jvm.javaMethod

/**
 * Translates a [DedaggedTracingTensor] into a JVM method of the signature ([FloatArray] -> [OutputType]).
 * Returns a function that invokes the generated JVM static method and reassembles the output object
 * based on the values computed in the jvm code.
 *
 * See also https://bytebuddy.net/ and https://asm.ow2.io/
 */
internal fun <OutputType : Any> jvmTracingTranslator(trace: DedaggedTracingTensor<OutputType>): ((FloatArray) -> OutputType)? {
    require(trace.canScalarEval)
    val jvmGenerate = JvmGenerator(trace) // generate the bytecode
    val fn = jvmGenerate.getEvaluator() ?: return null   // access the created class

    // produce a function that calls it and assembles the [OutputType] object.
    val resultingFunction: (FloatArray) -> OutputType = { input: FloatArray ->
        val neededArraySize = trace.numInputs + trace.numTemps + trace.numResults
        val variables: FloatArray = if (input.size < neededArraySize) Arrays.copyOf(input, neededArraySize) else input
        fn.invoke(variables) // run the generated bytecode
        var nextOutput = trace.numInputs + trace.numTemps
        val assembleOutput = object : Wrapper() {
            override fun wrapDTensor(value: DTensor): DTensor {
                return when (value) {
                    is TracingScalar -> FloatScalar(variables[nextOutput++])
                    else -> value
                }
            }

            override fun wrapRandomKey(value: RandomKey): RandomKey {
                TODO("Not yet implemented")
            }
        }
        val result = assembleOutput.wrap(trace.value)
        require(nextOutput == neededArraySize)
        result
    }

    return resultingFunction
}

/**
 * This abstract class is implemented by the generated code.
 * The generated code leaves the computed output results in the array.
 * The array is logically partitioned into three sections, based on
 * the values in [DedaggedTracingTensor].  First, there are
 * [DedaggedTracingTensor.numInputs] inputs that are filled in with values
 * from the input.  Then there are [DedaggedTracingTensor.numTemps]
 * places that can be used by the generated code to store intermediate
 * results.  Then there are [DedaggedTracingTensor.numResults] where the
 * generated code places the results, which will be reassembled by the
 * called into the appropriate output object type.
 */
internal abstract class Evaluator {
    abstract fun invoke(t: FloatArray)
}

/**
 * An instruction generator to produce code to evaluate a trace into the body of an override of
 * [Evaluator.invoke].
 */
private class InstructionGenerator(
    val result: MutableList<StackManipulation>,
    val trace: DedaggedTracingTensor<*>,
    val useLocals: Boolean = true,
) : TracingVisitor<Unit> {
    val traceId = trace.traceId
    val tempIndexRange = trace.numInputs until trace.numInputs + trace.numTemps
    val localOffset = 2 - trace.numInputs
    var numLocals = 2

    fun generateStore(variableIndex: Int, t: TracingTensor) {
        if (useLocals && variableIndex in tempIndexRange) {
            val localIndex = variableIndex + localOffset
            numLocals = max(numLocals, localIndex + 1)
            // compute the value to be stored
            this.visit(t)
            add(
                MethodVariableAccess.of(floatType).storeAt(localIndex)
            )
        } else {
            add(
                // Prepare to store at array index i
                variableArray(), // the function's FloatArray parameter
                IntegerConstant.forValue(variableIndex),
            )
            // compute the value to be stored
            this.visit(t)
            add(
                // store arg[i]
                ArrayAccess.FLOAT.store(),
            )
        }
    }

    fun generateLoad(variableIndex: Int) {
        if (useLocals && variableIndex in tempIndexRange) {
            val localIndex = variableIndex + localOffset
            add(
                MethodVariableAccess.of(floatType).loadFrom(localIndex)
            )
        } else {
            add(
                variableArray(),
                IntegerConstant.forValue(variableIndex),
                ArrayAccess.FLOAT.load(),
            )
        }
    }

    fun finish() {
        add(
            // return
            MethodReturn.VOID,
        )
    }

    private fun add(vararg s: StackManipulation) {
        result.addAll(s)
    }

    /**
     * Push onto the stack a reference to the reference parameter of the
     * method, which contains the float array for the variables.
     */
    private fun variableArray(): StackManipulation {
        return MethodVariableAccess.REFERENCE.loadFrom(1)
    }

    override fun visitConstant(x: TracingTensor.Constant) {
        require(x.values is FloatScalar)
        add(
            FloatConstant.forValue(x.values.value)
        )
    }

    override fun visitVariable(x: TracingTensor.Variable) {
        require(x.traceId == traceId)
        require(x.isScalar)
        generateLoad(x.varIndex)
    }

    override fun visitPlus(x: TracingTensor.Plus) {
        require(x.isScalar)
        visit(x.left)
        visit(x.right)
        add(
            Addition.FLOAT
        )
    }

    override fun visitMinus(x: TracingTensor.Minus) {
        require(x.isScalar)
        visit(x.left)
        visit(x.right)
        add(
            Subtraction.FLOAT
        )
    }

    override fun visitTimes(x: TracingTensor.Times) {
        visit(x.left)
        visit(x.right)
        require(x.isScalar)
        add(
            Multiplication.FLOAT
        )
    }

    override fun visitTimesScalar(x: TracingTensor.TimesScalar) {
        require(x.isScalar)
        visit(x.left)
        visit(x.right)
        add(
            Multiplication.FLOAT
        )
    }

    override fun visitDiv(x: TracingTensor.Div) {
        require(x.isScalar)
        visit(x.left)
        visit(x.right)
        add(
            Division.FLOAT
        )
    }

    override fun visitZero(x: TracingTensor.Zero) {
        require(x.isScalar)
        add(
            FloatConstant.ZERO
        )
    }

    override fun visitIdentityGradient(x: TracingTensor.IdentityGradient) {
        require(x.isScalar)
        add(
            FloatConstant.ONE
        )
    }

    companion object {
        val floatType = TypeDescription.ForLoadedType.of(FloatArray::class.java.componentType)

        internal open class MyStackManipulation(
            val opcode: Int,
            val sizeImpact: Int,
            val maximalSize: Int
        ) : StackManipulation {
            override fun isValid(): Boolean = true
            override fun apply(
                methodVisitor: MethodVisitor,
                implementationContext: Implementation.Context
            ): StackManipulation.Size {
                methodVisitor.visitInsn(opcode)
                return StackManipulation.Size(sizeImpact, maximalSize)
            }
        }

        object FNEG : MyStackManipulation(Opcodes.FNEG, 0, 1)
        object F2D : MyStackManipulation(Opcodes.F2D, 1, 2)
        object D2F : MyStackManipulation(Opcodes.D2F, -1, 2)
        object F2I : MyStackManipulation(Opcodes.F2I, 0, 1)
        object FCMPG : MyStackManipulation(Opcodes.FCMPG, -1, 2)
        object FCMPL : MyStackManipulation(Opcodes.FCMPL, -1, 2)
        internal open class LABEL(val label: Label) : StackManipulation {
            override fun isValid() = true
            override fun apply(
                methodVisitor: MethodVisitor,
                implementationContext: Implementation.Context
            ): StackManipulation.Size {
                methodVisitor.visitLabel(label)
                return StackManipulation.Size(0, 0)
            }
        }
        internal open class Branch(val opcode: Int, val label: Label) : StackManipulation {
            override fun isValid() = true
            override fun apply(
                methodVisitor: MethodVisitor,
                implementationContext: Implementation.Context
            ): StackManipulation.Size {
                methodVisitor.visitJumpInsn(opcode, label)
                return StackManipulation.Size(-1, 1)
            }
        }
        class IFGT(label: Label) : Branch(Opcodes.IFGT, label)
        class IFGE(label: Label) : Branch(Opcodes.IFGE, label)
        class IFLT(label: Label) : Branch(Opcodes.IFLT, label)
        class IFLE(label: Label) : Branch(Opcodes.IFLE, label)
        class IFEQ(label: Label) : Branch(Opcodes.IFEQ, label)
        class IFNE(label: Label) : Branch(Opcodes.IFNE, label)
        class GOTO(label: Label) : Branch(Opcodes.GOTO, label)
        object RELU : StackManipulation {
            override fun isValid(): Boolean = true
            override fun apply(
                methodVisitor: MethodVisitor,
                implementationContext: Implementation.Context
            ): StackManipulation.Size {
                val endOfBlockLabel = Label()
                methodVisitor.visitInsn(Opcodes.DUP)
                methodVisitor.visitInsn(Opcodes.FCONST_0)
                methodVisitor.visitInsn(Opcodes.FCMPG) // 1 if NaN or > 0
                methodVisitor.visitJumpInsn(Opcodes.IFGT, endOfBlockLabel)
                methodVisitor.visitInsn(Opcodes.POP)
                methodVisitor.visitInsn(Opcodes.FCONST_0)
                methodVisitor.visitLabel(endOfBlockLabel)
                return StackManipulation.Size(0, 3) // 3 or 2
            }
        }
        object RELUGRAD : StackManipulation {
            override fun isValid(): Boolean = true
            override fun apply(
                methodVisitor: MethodVisitor,
                implementationContext: Implementation.Context
            ): StackManipulation.Size {
                val endOfBlockLabel = Label()
                // Expects stack to be [upstream, x]
                methodVisitor.visitInsn(Opcodes.FCONST_0)
                methodVisitor.visitInsn(Opcodes.FCMPG)
                methodVisitor.visitJumpInsn(Opcodes.IFGT, endOfBlockLabel)
                methodVisitor.visitInsn(Opcodes.POP) // pop upstream and put 0f
                methodVisitor.visitInsn(Opcodes.FCONST_0)
                methodVisitor.visitLabel(endOfBlockLabel)
                return StackManipulation.Size(-1, 2)
            }
        }
    }

    override fun visitUnaryMinus(x: TracingTensor.UnaryMinus) {
        require(x.isScalar)
        visit(x.x)
        add(
            FNEG
        )
    }

    override fun visitMatmul(x: TracingTensor.Matmul) = throw IllegalStateException("Not scalar")
    override fun visitOuterProduct(x: TracingTensor.OuterProduct) = throw IllegalStateException("Not scalar")

    override fun visitSin(x: TracingTensor.Sin) {
        require(x.isScalar)
        visit(x.x)
        add(
            F2D,
            MethodInvocation.invoke(
                MethodDescription.ForLoadedMethod(
                    java.lang.Math::sin.javaMethod!!
                )
            ),
            D2F
        )
    }

    override fun visitCos(x: TracingTensor.Cos) {
        require(x.isScalar)
        visit(x.x)
        add(
            F2D,
            MethodInvocation.invoke(
                MethodDescription.ForLoadedMethod(
                    java.lang.Math::cos.javaMethod!!
                )
            ),
            D2F
        )
    }

    override fun visitTan(x: TracingTensor.Tan) {
        require(x.isScalar)
        visit(x.x)
        add(
            F2D,
            MethodInvocation.invoke(
                MethodDescription.ForLoadedMethod(
                    java.lang.Math::tan.javaMethod!!
                )
            ),
            D2F
        )
    }

    override fun visitAtan(x: TracingTensor.Atan) {
        require(x.isScalar)
        visit(x.x)
        add(
            F2D,
            MethodInvocation.invoke(
                MethodDescription.ForLoadedMethod(
                    java.lang.Math::atan.javaMethod!!
                )
            ),
            D2F
        )
    }

    override fun visitExp(x: TracingTensor.Exp) {
        require(x.isScalar)
        visit(x.x)
        add(
            F2D,
            MethodInvocation.invoke(
                MethodDescription.ForLoadedMethod(
                    java.lang.Math::exp.javaMethod!!
                )
            ),
            D2F
        )
    }

    override fun visitLn(x: TracingTensor.Ln) {
        require(x.isScalar)
        visit(x.x)
        add(
            F2D,
            MethodInvocation.invoke(
                MethodDescription.ForLoadedMethod(
                    java.lang.Math::log.javaMethod!!
                )
            ),
            D2F
        )
    }

    override fun visitLgamma(x: TracingTensor.Lgamma) {
        require(x.isScalar)
        visit(x.x)
        val lgamma: KFunction1<Float, Float> = External::lgamma
        add(
            MethodInvocation.invoke(
                MethodDescription.ForLoadedMethod(
                    lgamma.javaMethod!!
                )
            )
        )
    }

    override fun visitDigamma(x: TracingTensor.Digamma) {
        require(x.isScalar)
        visit(x.x)
        val digamma: KFunction1<Float, Float> = External::digamma
        add(
            MethodInvocation.invoke(
                MethodDescription.ForLoadedMethod(
                    digamma.javaMethod!!
                )
            )
        )
    }

    override fun visitPolygamma(x: TracingTensor.Polygamma) {
        require(x.isScalar)
        add(
            IntegerConstant.forValue(x.n)
        )
        visit(x.x)
        val polygamma: KFunction2<Int, Float, Float> = External::polygamma
        add(
            MethodInvocation.invoke(
                MethodDescription.ForLoadedMethod(
                    polygamma.javaMethod!!
                )
            )
        )
    }

    override fun visitSqrt(x: TracingTensor.Sqrt) {
        require(x.isScalar)
        visit(x.x)
        add(
            F2D,
            MethodInvocation.invoke(
                MethodDescription.ForLoadedMethod(
                    java.lang.Math::sqrt.javaMethod!!
                )
            ),
            D2F
        )
    }

    override fun visitTanh(x: TracingTensor.Tanh) {
        require(x.isScalar)
        visit(x.x)
        add(
            F2D,
            MethodInvocation.invoke(
                MethodDescription.ForLoadedMethod(
                    java.lang.Math::tanh.javaMethod!!
                )
            ),
            D2F
        )
    }

    override fun visitMeld(x: TracingTensor.Meld) = throw IllegalStateException("Not scalar")
    override fun visitSplit(x: TracingTensor.Split) = throw IllegalStateException("Not scalar")
    override fun visitSplitPart(x: TracingTensor.SplitPart) = throw IllegalStateException("Not scalar")
    override fun visitConcat(x: TracingTensor.Concat) = throw IllegalStateException("Not scalar")
    override fun visitBroadcastTo(x: TracingTensor.BroadcastTo) = throw IllegalStateException("Not scalar")
    override fun visitConvImpl(x: TracingTensor.ConvImpl) = throw IllegalStateException("Not scalar")
    override fun visitExpand(x: TracingTensor.Expand) = throw IllegalStateException("Not scalar")
    override fun visitFlip(x: TracingTensor.Flip) = throw IllegalStateException("Not scalar")
    override fun visitLogSoftmax(x: TracingTensor.LogSoftmax) = throw IllegalStateException("Not scalar")
    override fun visitLogSoftmaxGrad(x: TracingTensor.LogSoftmaxGrad) = throw IllegalStateException("Not scalar")

    override fun visitPow(x: TracingTensor.Pow) {
        require(x.isScalar)
        visit(x.base)
        add(
            F2D,
            FloatConstant.forValue(x.exponent),
            F2D,
            MethodInvocation.invoke(
                MethodDescription.ForLoadedMethod(
                    java.lang.Math::pow.javaMethod!!
                )
            ),
            D2F
        )
    }

    override fun visitView1(x: TracingTensor.View1) = throw IllegalStateException("Not scalar")
    override fun visitView2(x: TracingTensor.View2) = throw IllegalStateException("Not scalar")
    override fun visitView3(x: TracingTensor.View3) = throw IllegalStateException("Not scalar")
    override fun visitReshape(x: TracingTensor.Reshape) = throw IllegalStateException("Not scalar")
    override fun visitReshapeToScalar(x: TracingScalar.ReshapeToScalar) = throw IllegalStateException("Not scalar")
    override fun visitSqueeze(x: TracingTensor.Squeeze) = throw IllegalStateException("Not scalar")
    override fun visitUnsqueeze(x: TracingTensor.Unsqueeze) = throw IllegalStateException("Not scalar")
    override fun visitTranspose(x: TracingTensor.Transpose) = throw IllegalStateException("Not scalar")

    override fun visitRelu(x: TracingTensor.Relu) {
        require(x.isScalar)
        visit(x.x)
        add(
            RELU
        )
    }

    override fun visitReluGrad(x: TracingTensor.ReluGrad) {
        require(x.isScalar)
        // We visit in this particular order so the operand stack will be [upstream, x]
        visit(x.upstream)
        visit(x.x)
        add(
            RELUGRAD
        )
    }

    override fun visitSigmoid(x: TracingTensor.Sigmoid) {
        require(x.isScalar)
        visit(x.x)
        add(
            MethodInvocation.invoke(
                MethodDescription.ForLoadedMethod(
                    ::sigmoidElem.javaMethod!!
                )
            )
        )
    }

    // TODO (#181): Translate tracing random floats and keys
    override fun visitRandomFloats(x: TracingTensor.RandomFloats) = TODO("Not yet implemented")
    override fun visitRandomVariable(x: TracingRandomKey.Variable) = TODO("Not yet implemented")
    override fun visitRandomSplitPart(x: TracingRandomKey.SplitPart) = TODO("Not yet implemented")
    override fun visitRandomSplit(x: TracingRandomKey.Split) = TODO("Not yet implemented")

    override fun visitSum(x: TracingTensor.Sum) = throw IllegalStateException("Not scalar")
    override fun visitAvgPool(x: TracingTensor.AvgPool) = throw IllegalStateException("Not scalar")
    override fun visitAvgPoolGrad(x: TracingTensor.AvgPoolGrad) = throw IllegalStateException("Not scalar")
    override fun visitMaxPoolWithIndices(x: TracingTensor.MaxPoolWithIndices) =
        throw IllegalStateException("Not scalar")

    override fun visitGather(x: TracingTensor.Gather) = throw IllegalStateException("Not scalar")
    override fun visitGatherAtIndices(x: TracingTensor.GatherAtIndices) = throw IllegalStateException("Not scalar")
    override fun visitScatter(x: TracingTensor.Scatter) = throw IllegalStateException("Not scalar")
    override fun visitScatterAtIndices(x: TracingTensor.ScatterAtIndices) = throw IllegalStateException("Not scalar")

    fun conditionalGoto(kind: ComparisonKind, label: Label, branchOnNan: Boolean = true) {
        when (kind) {
            ComparisonKind.GT -> add(if (branchOnNan) FCMPG else FCMPL, IFGT(label))
            ComparisonKind.GE -> add(if (branchOnNan) FCMPG else FCMPL, IFGE(label))
            ComparisonKind.LT -> add(if (branchOnNan) FCMPL else FCMPG, IFLT(label))
            ComparisonKind.LE -> add(if (branchOnNan) FCMPL else FCMPG, IFLE(label))
            ComparisonKind.EQ -> add(FCMPL, IFEQ(label))
            ComparisonKind.NE -> add(FCMPL, IFNE(label))
        }
    }

    override fun visitCompare(x: TracingTensor.Compare) {
        require(x is TracingScalar)
        // TODO: Should we consider using arithmetic to compute the result rather
        // than control-flow?  We could if we don't restrict the result to 0 or 1
        // For example, to test
        // if A is greater than B, return A-B.
        // if A is less than B, return B-A.
        // if A is greater than or equal to B, return A-B+Float.MIN_VALUE.
        // if A is less than or equal to B, return B-A+Float.MIN_VALUE.
        // EQ and NE are left as an exercise for the reader
        // This works for scalars and tensors

        val falseLabel = Label()
        val endOfBlockLabel = Label()

        goIfFalse(x, falseLabel)
        add(
            FloatConstant.forValue(1f),
            GOTO(endOfBlockLabel),
            LABEL(falseLabel),
            FloatConstant.forValue(0f),
            LABEL(endOfBlockLabel)
        )
    }

    fun goIfFalse(cond: TracingScalar, label: Label) {
        if (cond is TracingTensor.Compare) {
            visit(cond.left)
            visit(cond.right)
            conditionalGoto(cond.comparison.inverted, label)
        } else {
            visit(cond)
            add(FloatConstant.forValue(0f))
            conditionalGoto(ComparisonKind.GT.inverted, label)
        }
    }

    override fun visitIfThenElse(x: TracingTensor.IfThenElse) {
        require(x.cond is TracingScalar)
        val falseLabel = Label()
        val endOfBlockLabel = Label()

        goIfFalse(x.cond, falseLabel)
        visit(x.whenTrue)
        add(
            GOTO(endOfBlockLabel),
            LABEL(falseLabel)
        )
        visit(x.whenFalse)
        add(LABEL(endOfBlockLabel))
    }
}

private class GeneratedImplementation<OutputType : Any>(val trace: DedaggedTracingTensor<OutputType>) : Implementation {
    override fun prepare(instrumentedType: InstrumentedType): InstrumentedType {
        return instrumentedType
    }

    override fun appender(implementationTarget: Implementation.Target): ByteCodeAppender {
        return object : ByteCodeAppender {
            override fun apply(
                methodVisitor: MethodVisitor,
                implementationContext: Implementation.Context,
                instrumentedMethod: MethodDescription
            ): ByteCodeAppender.Size {
                val (stackOperations, numLocals) = makeStackManipulations()
                val operandStackSize = StackManipulation.Compound(
                    *stackOperations.toTypedArray()
                ).apply(methodVisitor, implementationContext)
                return ByteCodeAppender.Size(
                    operandStackSize.maximalSize,
                    numLocals
                )
            }

            fun makeStackManipulations(): Pair<List<StackManipulation>, Int> {
                val result = mutableListOf<StackManipulation>()
                val gen = InstructionGenerator(result, trace)
                for ((tempId, tempTrace) in trace.assignments) {
                    require(tempTrace is TracingTensor)
                    gen.generateStore(tempId, tempTrace)
                }
                var nextOutput = trace.numInputs + trace.numTemps
                val wrapper = object : Wrapper() {
                    override fun wrapDTensor(value: DTensor): DTensor {
                        if (value is TracingTensor)
                            gen.generateStore(nextOutput++, value)
                        return value
                    }

                    override fun wrapRandomKey(value: RandomKey): RandomKey {
                        TODO("Not yet implemented")
                    }
                }
                wrapper.wrap(trace.value)
                require(trace.numInputs + trace.numTemps + trace.numResults == nextOutput)
                gen.finish()
                val localsSize = gen.numLocals
                return Pair(result, localsSize)
            }
        }
    }
}


internal object EnableComputeFrames : AsmVisitorWrapper {
    override fun mergeWriter(flags: Int): Int = flags or ClassWriter.COMPUTE_FRAMES

    override fun mergeReader(flags: Int): Int = flags

    override fun wrap(
        instrumentedType: TypeDescription,
        classVisitor: ClassVisitor,
        implementationContext: Implementation.Context,
        typePool: TypePool,
        fields: FieldList<FieldDescription.InDefinedShape>,
        methods: MethodList<*>,
        writerFlags: Int,
        readerFlags: Int
    ): ClassVisitor = classVisitor
}

/**
 * A visitor that generates the bytecode necessary to evaluate a given tracing tensor to the stack.
 */
internal class JvmGenerator<OutputType : Any>(val trace: DedaggedTracingTensor<OutputType>) {
    private val implementation = GeneratedImplementation<OutputType>(trace)
    private val meth = ByteBuddy().subclass(Evaluator::class.java)
        .visit(EnableComputeFrames)
        .method(ElementMatchers.named("invoke"))
    private val instance = try {
        val l = meth.intercept(implementation).make()
            .load(javaClass.classLoader)
            .loaded
        l.newInstance()
    } catch (x: MethodTooLargeException) {
        null
    }

    fun getEvaluator(): Evaluator? {
        return instance
    }
}
