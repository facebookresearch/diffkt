/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "util.h"
#include <cmath>
#include <cfloat>
#include <limits>
#include <cstdint>


namespace math { namespace util {
using namespace std;

inline double polevl(double x, double *A, size_t len) {
  double result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + A[i];
  }
  return result;
}

/*
 * The Hurwitz Zeta function taken from Aten which was derived from
 * the Cephes Math Library
 */
inline double zeta(double x, double q) {
  static double MACHEP = 1.11022302462515654042E-16;
  static double A[] = {
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.8924375803183791606e9, /*1.307674368e12/691*/
    7.47242496e10,
    -2.950130727918164224e12, /*1.067062284288e16/3617*/
    1.1646782814350067249e14, /*5.109094217170944e18/43867*/
    -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
    1.8152105401943546773e17, /*1.5511210043330985984e23/854513*/
    -7.1661652561756670113e18 /*1.6938241367317436694528e27/236364091*/
  };

  int i = 0;
  double a, b, k, s, t, w;
  if (x == 1.0) {
    return INFINITY;
  }

  if (x < 1.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  if (q <= 0.0) {
    if (q == floor(q)) {
      return INFINITY;
    }
    if (x != floor(x)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  s = std::pow(q, -x);
  a = q;
  i = 0;
  b = 0.0;
  while ((i < 9) || (a <= 9.0)) {
    i += 1;
    a += 1.0;
    b = std::pow(a, -x);
    s += b;
    if ((-MACHEP * s < b) && (b < MACHEP * s)) {
      return s;
    }
  };

  w = a;
  s += b * w / (x - 1.0);
  s -= 0.5 * b;
  a = 1.0;
  k = 0.0;
  for (int i = 0; i < 12; i++) {
    a *= x + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    t = std::abs(t / s);
    if (t < MACHEP) {
      return s;
    }
    k += 1.0;
    a *= x + k;
    b /= w;
    k += 1.0;
  }
  return s;
}

/*
 * The digamma function taken from Aten which was derived from
 * the Cephes Math Library
 */
double digamma(double x) {
  // [C++ Standard Reference: Gamma Function] https://en.cppreference.com/w/cpp/numeric/math/tgamma
  static double PSI_10 = 2.25175258906672110764;
  if (x == 0) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    return std::copysign(INFINITY, -x);
  }

  bool x_is_integer = x == trunc(x);
  if (x < 0) {
    if (x_is_integer) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is a negative integer, NaN is returned
      return std::numeric_limits<double>::quiet_NaN();
    }
    // Extracts the fractional part of x as r, since tan(pi * r) is more numerically
    // accurate than tan(pi * x). While these operations are mathematically equivalent
    // since both x and r are in radians and tan() has a periodicity of pi, in practice
    // the computation of pi * x is a source of error (when |x| > 1).
    double q, r;
    r = std::modf(x, &q);
    return digamma(1 - x) - M_PI / tan(M_PI * r);
  }

  // Push x to be >= 10
  double result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    return result + PSI_10;
  }

  // Compute asymptotic digamma
  static double A[] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  double y = 0;
  if (x < 1.0e17) {
    double z = 1.0 / (x * x);
    y = z * polevl(z, A, 6);
  }
  return result + log(x) - (0.5 / x) - y;
}

/*
 * The trigamma function taken from Aten
 */
double trigamma(double x) {
  double sign = +1;
  double result = 0;
  if (x < 0.5) {
    sign = -1;
    const double sin_pi_x = sin(M_PI * x);
    result -= (M_PI * M_PI) / (sin_pi_x * sin_pi_x);
    x = 1 - x;
  }
  for (int i = 0; i < 6; ++i) {
    result += 1 / (x * x);
    x += 1;
  }
  const double ixx = 1 / (x*x);
  result += (1 + 1 / (2*x) + ixx * (1./6 - ixx * (1./30 - ixx * (1./42)))) / x;
  return sign * result;
}
/*
 * The polygamma function
 * call digamma or trigamma for n=0 or n=1 respectively
 */
double polygamma(int64_t n, double x) {
  return ((n % 2) ? 1.0 : -1.0) * std::exp(lgamma(double(n) + 1.0)) *
      zeta(double(n + 1), x);
}

}} // namespace math::util
