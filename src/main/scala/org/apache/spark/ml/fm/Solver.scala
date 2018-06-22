package org.apache.spark.ml.fm

import scala.{specialized => spec}
import scala.reflect._

class SolverAggregator[@spec(Float, Double) V: Fractional : ClassTag](k: Int) extends Serializable {
  require(k > 0)

  val fracV = implicitly[Fractional[V]]

  import fracV._

  // first k elements store X*y, the following k * (k + 1) / 2 elements store Xi*Xj
  val stat = Array.fill(k * (k + 3) / 2)(zero)

  def update(w: V,
             x: Array[V],
             y: V): SolverAggregator[V] = {
    var index = k
    var i = 0
    var j = 0
    while (i < k) {
      j = i
      while (j < k) {
        stat(index) += x(i) * x(j) * w
        index += 1
        j += 1
      }
      stat(i) += x(i) * y * w
      i += 1
    }
    this
  }

  def merge(o: SolverAggregator[V]): SolverAggregator[V] = {
    var i = 0
    while (i < stat.length) {
      stat(i) += o.stat(i)
      i += 1
    }
    this
  }

  def solve(regL1: V,
            regL2: V,
            regLG: V,
            prev: Array[V],
            iters: Int): Array[V] = {
    Solver.solve(stat, k, regL1, regL2, regLG, prev, iters)
  }
}


object Solver extends Serializable {
  def initStat[@spec(Float, Double) V: Fractional : ClassTag](k: Int): Array[V] = {
    // first k elements store X*y, the following k * (k + 1) / 2 elements store Xi*Xj
    val fracV = implicitly[Fractional[V]]
    import fracV._

    Array.fill(k * (k + 3) / 2)(zero)
  }

  def updateStat[@spec(Float, Double) V: Fractional : ClassTag](stat: Array[V],
                                                                k: Int,
                                                                w: V,
                                                                x: Array[V],
                                                                y: V): Array[V] = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    var index = k
    var i = 0
    var j = 0
    while (i < k) {
      j = i
      while (j < k) {
        stat(index) += x(i) * x(j) * w
        index += 1
        j += 1
      }
      stat(i) += x(i) * y * w
      i += 1
    }

    stat
  }

  def mergeStat[@spec(Float, Double) V: Fractional : ClassTag](stat1: Array[V],
                                                               stat2: Array[V]): Array[V] = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    require(stat1.length == stat2.length)
    var i = 0
    while (i < stat1.length) {
      stat1(i) += stat2(i)
      i += 1
    }

    stat1
  }

  def solve[@spec(Float, Double) V: Fractional : ClassTag](stat: Array[V],
                                                           k: Int,
                                                           regL1: V,
                                                           regL2: V,
                                                           regLG: V,
                                                           prev: Array[V],
                                                           iters: Int): Array[V] = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    val solution = prev.clone()

    for (iter <- 0 until iters; i <- 0 until k) {
      val a = getXiXj(stat, i, i, k) + regL2

      var b = getXiY(stat, i)
      for (j <- 0 until k if j != i) {
        b -= solution(j) * getXiXj(stat, i, j, k)
      }
      b = -(b + b)

      if (regLG == zero ||
        solution.zipWithIndex.filter(_._2 != k).forall(_._1 == zero)) {
        // standard one-variable elastic net problem
        val c = regL1 + regLG
        val s = solveEN(a, b, c)
        if (s.nonEmpty) {
          solution(i) = s.get
        }

      } else {

        val w0 = solution(i)
        val e = solution.map(w => w * w).sum - w0 * w0
        val s = solveGL(a, b, regL1, regLG, e, w0)
        if (s.nonEmpty) {
          solution(i) = s.get
        }
      }
    }

    solution
  }


  private def getXiXj[@spec(Float, Double) V](stat: Array[V],
                                              i: Int,
                                              j: Int,
                                              k: Int): V = {
    val index = if (i < j) {
      k + (k * 2 - i - 1) * i / 2 + j
    } else {
      k + (k * 2 - j - 1) * j / 2 + i
    }
    stat(index)
  }

  private def getXiY[@spec(Float, Double) V](stat: Array[V], i: Int): V = {
    stat(i)
  }


  // f(x) = a x^2 + b x + c |x|
  // return argmin_x f(x)
  private def solveEN[@spec(Float, Double) V: Fractional](a: V,
                                                          b: V,
                                                          c: V): Option[V] = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    if (a > zero && c >= zero) {
      val s = if (b + c < zero) {
        -(b + c) / a / fromInt(2)
      } else if (b - c > zero) {
        (c - b) / a / fromInt(2)
      } else {
        zero
      }
      Some(s)
    } else {
      None
    }
  }


  // f(x) = a x^2 + b x + c |x| + d (x^2 + e)^0.5
  // return argmin_x f(x)
  private def solveGL[@spec(Float, Double) V: Fractional](a: V,
                                                          b: V,
                                                          c: V,
                                                          d: V,
                                                          e: V,
                                                          x0: V,
                                                          iters: Int = 5): Option[V] = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    // use two order approx near x0 instead of d (x^2 + e)^0.5

    if (a > zero && c >= zero && d >= zero && e >= zero) {

      val n = sqrt(x0 * x0 + e)
      val g1 = d * x0 / n
      val g2 = d * e / n / n / n

      val a_ = a + g2 / fromInt(2)
      val b_ = b + g1 + g2 * x0

      val l0 = loss(a, b, c, d, e, x0)

      val s0 = solveEN(a_, b_, c)

      if (s0.isEmpty) {
        None
      } else {
        var ok = false
        val decay = fromInt(618) / fromInt(1000)
        var v = s0.get - x0
        for (i <- 0 until iters) {
          if (!ok) {
            if (loss(a, b, c, d, e, x0 + v) < l0) {
              ok = true
            } else {
              v *= decay
            }
          }
        }

        if (ok) {
          Some(x0 + v)
        } else {
          None
        }
      }
    } else {
      None
    }
  }


  // return a x^2 + b x + c |x| + d (x^2 + e)^0.5
  private def loss[@spec(Float, Double) V: Fractional](a: V,
                                                       b: V,
                                                       c: V,
                                                       d: V,
                                                       e: V,
                                                       x: V): V = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    a * x * x +
      b * x +
      c * abs(x) +
      d * sqrt(x * x + e)
  }


  private def sqrt[@spec(Float, Double) V: Fractional](value: V): V = {
    value match {
      case v: Float =>
        math.sqrt(v).toFloat.asInstanceOf[V]

      case v: Double =>
        math.sqrt(v).asInstanceOf[V]
    }
  }
}
