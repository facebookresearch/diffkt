package examples
import java.net.URL
import org.diffkt.*

fun main() {

    // Import golden retriever weights
    val allData = URL("https://bit.ly/3ILpk6E")
        .readText().split(Regex("\\r?\\n"))
        .filter { it.isNotBlank() }
        .map { it.toFloat() }

    // Create tensor off golden retriever weights
    val dataTensor = allData.map(::FloatScalar).let(::tensorOf)

    // Initialize mean and standard deviation
    // at one of the data points
    var mean: DScalar = FloatScalar(allData[0])
    var stdDev: DScalar = FloatScalar(1F)

    // normal distribution function
    fun normalDistribution(x: DTensor, mean: DScalar, stdDev: DScalar) =
        1.0F / (stdDev*(2.0F * FloatScalar.PI).pow(.5F)) * exp(-.5F * ((x - mean) / stdDev).pow(2))

    fun mle(mean: DScalar, stdDev: DScalar) = ln(normalDistribution(dataTensor, mean, stdDev)).sum()

    // perform gradient descent
    val L = .01F
    val iterations = 100_000

    for (i in 0..iterations) {
        val (meanGradient, stdDevGradient) = forwardDerivatives(mean, stdDev, ::mle)

        mean += L * meanGradient
        stdDev += L * stdDevGradient
    }

    print("mean=$mean, stdDev=$stdDev")
}
