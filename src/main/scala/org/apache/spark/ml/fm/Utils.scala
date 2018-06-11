package org.apache.spark.ml.fm

import java.{util => ju}

import scala.reflect.{ClassTag, classTag}
import scala.{specialized => spec}

import scala.collection.mutable
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.reflect.{ClassTag, classTag}
import scala.util.{Failure, Random, Success}
import org.apache.hadoop.fs.Path
import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.XORShiftRandom

private trait FromDouble[H] extends Serializable {

  def fromDouble(value: Double): H
}


private object DoubleFromDouble extends FromDouble[Double] {

  override def fromDouble(value: Double): Double = value
}


private object FloatFromDouble extends FromDouble[Float] {

  override def fromDouble(value: Double): Float = value.toFloat
}


private object DecimalFromDouble extends FromDouble[BigDecimal] {

  override def fromDouble(value: Double): BigDecimal = BigDecimal(value)
}


private[fm] object FromDouble {

  implicit final val doubleFromDouble: FromDouble[Double] = DoubleFromDouble

  implicit final val floatFromDouble: FromDouble[Float] = FloatFromDouble

  implicit final val decimalFromDouble: FromDouble[BigDecimal] = DecimalFromDouble
}


private[fm] object Utils extends Logging {

  val INSTANCE_INDEX = "instance_index"
  val INSTANCE_LABEL = "instance_label"
  val INSTANCE_WEIGHT = "instance_weight"

  val INDEX = "index"
  val GROUP = "group"
  val INTRA_INDEX = "intra_index"
  val INDICES = "indices"
  val VALUES = "values"
  val VALUE = "value"

  val ERROR = "error"
  val PREDICTION = "prediction"
  val DOTS = "dots"
  val PREDICTION_DOTS = "prediction_dots"

  val EH = "eh"

  val WEIGHT = "weight"
  val FACTOR = "factor"

  val PREVIOUS_WEIGHT = "previous_weight"
  val PREVIOUS_FACTOR = "previous_factor"
  val PREVIOUS_EH = "previous_eh"

  val PROBLEM_YX = "problem_yx"

  val STAT = "stat"
  val SOLUTION = "solution"
  val PREV_SOLUTION = "prev_solution"

  val RANDOM = "random"
  val SELECTED = "selected"
  val SELECTED_INDICES = "selected_indices"

  val FIRST = "first"


  def initStat[@spec(Float, Double) V: Fractional : ClassTag](k: Int): Array[V] = {
    // first k elements store X*y, the following k * (k + 1) / 2 elements store Xi*Xj
    Array.ofDim[V](k * (k + 3) / 2)
  }

  def updateStat[@spec(Float, Double) V: Fractional](stat: Array[V],
                                                     k: Int,
                                                     w: V,
                                                     x: Array[V],
                                                     y: V): Unit = {
    val fraV = implicitly[Fractional[V]]
    import fraV._

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
  }

  def mergeStat[@spec(Float, Double) V: Fractional](stat1: Array[V],
                                                    stat2: Array[V]): Unit = {
    val fraV = implicitly[Fractional[V]]
    import fraV._

    require(stat1.length == stat2.length)
    var i = 0
    while (i < stat1.length) {
      stat1(i) += stat2(i)
      i += 1
    }
  }

  def solve[@spec(Float, Double) V: Fractional : FromDouble : ClassTag](stat: Array[V],
                                                                        k: Int,
                                                                        regL1: Double,
                                                                        regL2: Double,
                                                                        regLG: Double,
                                                                        prev: Array[V],
                                                                        iters: Int): Array[V] = {
    val fraV = implicitly[Fractional[V]]
    import fraV._

    val stat_ = stat.map(_.toDouble)
    val prev_ = prev.map(_.toDouble)
    val solution = solveImpl(stat_, k, regL1, regL2, regLG, prev_, iters)

    val toV = implicitly[FromDouble[V]]
    solution.map(toV.fromDouble)
  }

  private def solveImpl(stat: Array[Double],
                        k: Int,
                        regL1: Double,
                        regL2: Double,
                        regLG: Double,
                        prev: Array[Double],
                        iters: Int): Array[Double] = {

    val solution = prev.clone()

    for (iter <- 0 until iters; i <- 0 until k) {
      val a = getXiXj(stat, i, i, k) + regL2

      var b = getXiY(stat, i)
      for (j <- 0 until k if j != i) {
        b -= solution(j) * getXiXj(stat, i, j, k)
      }
      b *= -2

      if (regLG == 0 ||
        solution.zipWithIndex.filter(_._2 != k).forall(_._1 == 0)) {
        // standard one-variable elastic net problem
        val c = regL1 + regLG
        val s = solveEN(a, b, c)
        if (!s.isNaN) {
          solution(i) = s
        }

      } else {

        val w0 = solution(i)
        val e = solution.map(w => w * w).sum - w0 * w0
        val s = solveGL(a, b, regL1, regLG, e, w0)
        if (!s.isNaN) {
          solution(i) = s
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
  private def solveEN(a: Double,
                      b: Double,
                      c: Double): Double = {
    if (a > 0 && c >= 0) {
      if (b + c < 0) {
        -(b + c) / a / 2
      } else if (b - c > 0) {
        (c - b) / a / 2
      } else {
        0.0F
      }
    } else {
      Double.NaN
    }
  }

  // return a x^2 + b x + c |x| + d (x^2 + e)^0.5
  private def loss(a: Double,
                   b: Double,
                   c: Double,
                   d: Double,
                   e: Double,
                   x: Double): Double = {
    a * x * x + b * x + c * math.abs(x) + d * math.sqrt(x * x + e)
  }

  // f(x) = a x^2 + b x + c |x| + d (x^2 + e)^0.5
  // return argmin_x f(x)
  private def solveGL(a: Double,
                      b: Double,
                      c: Double,
                      d: Double,
                      e: Double,
                      x0: Double,
                      iters: Int = 5): Double = {
    // use two order approx near x0 instead of d (x^2 + e)^0.5

    if (a > 0 && c >= 0 && d >= 0 && e >= 0) {
      val n = math.sqrt(x0 * x0 + e)
      val g1 = d * x0 / n
      val g2 = d * e / n / n / n

      val a_ = a + g2 / 2
      val b_ = b + g1 + g2 * x0

      val l0 = loss(a, b, c, d, e, x0)

      val s0 = solveEN(a_, b_, c)

      if (s0.isNaN) {
        Double.NaN
      } else {
        var ok = false
        var v = s0 - x0
        for (i <- 0 until iters) {
          if (!ok) {
            if (loss(a, b, c, d, e, x0 + v) < l0) {
              ok = true
            } else {
              v *= 0.618
            }
          }
        }

        if (ok) {
          x0 + v
        } else {
          Double.NaN
        }
      }
    } else {
      Double.NaN
    }
  }
}


/**
  * This class helps with persisting and checkpointing RDDs.
  *
  * Specifically, this abstraction automatically handles persisting and (optionally) checkpointing,
  * as well as unpersisting and removing checkpoint files.
  *
  * Users should call update() when a new Dataset has been created,
  * before the Dataset has been materialized.  After updating [[Checkpointer]], users are
  * responsible for materializing the Dataset to ensure that persisting and checkpointing actually
  * occur.
  *
  * When update() is called, this does the following:
  *  - Persist new Dataset (if not yet persisted), and put in queue of persisted Datasets.
  *  - Unpersist Datasets from queue until there are at most 2 persisted Datasets.
  *  - If using checkpointing and the checkpoint interval has been reached,
  *     - Checkpoint the new Dataset, and put in a queue of checkpointed Datasets.
  *     - Remove older checkpoints.
  *
  * WARNINGS:
  *  - This class should NOT be copied (since copies may conflict on which Datasets should be
  * checkpointed).
  *  - This class removes checkpoint files once later Datasets have been checkpointed.
  * However, references to the older Datasets will still return isCheckpointed = true.
  *
  * @param sc                 SparkContext for the Datasets given to this checkpointer
  * @param checkpointInterval Datasets will be checkpointed at this interval.
  *                           If this interval was set as -1, then checkpointing will be disabled.
  * @param storageLevel       caching storageLevel
  * @tparam T Dataset type, such as Double
  */
private[fm] class Checkpointer[T](val sc: SparkContext,
                                  val checkpointInterval: Int,
                                  val storageLevel: StorageLevel,
                                  val maxPersisted: Int) extends Logging {
  def this(sc: SparkContext, checkpointInterval: Int, storageLevel: StorageLevel) =
    this(sc, checkpointInterval, storageLevel, 2)

  require(storageLevel != StorageLevel.NONE)
  require(maxPersisted > 1)

  /** FIFO queue of past checkpointed Datasets */
  private val checkpointQueue = mutable.Queue.empty[RDD[T]]

  /** FIFO queue of past persisted Datasets */
  private val persistedQueue = mutable.Queue.empty[RDD[T]]

  /** Number of times [[update()]] has been called */
  private var updateCount = 0

  /**
    * Update with a new Dataset. Handle persistence and checkpointing as needed.
    * Since this handles persistence and checkpointing, this should be called before the Dataset
    * has been materialized.
    *
    * @param data New Dataset created from previous Datasets in the lineage.
    */
  def update(data: RDD[T]): Unit = {
    persist(data)
    persistedQueue.enqueue(data)
    while (persistedQueue.length > maxPersisted) {
      unpersist(persistedQueue.dequeue)
    }
    updateCount += 1

    // Handle checkpointing (after persisting)
    if (checkpointInterval != -1 && (updateCount % checkpointInterval) == 0
      && sc.getCheckpointDir.nonEmpty) {
      // Add new checkpoint before removing old checkpoints.
      checkpoint(data)
      checkpointQueue.enqueue(data)
      // Remove checkpoints before the latest one.
      var canDelete = true
      while (checkpointQueue.length > 1 && canDelete) {
        // Delete the oldest checkpoint only if the next checkpoint exists.
        if (isCheckpointed(checkpointQueue.head)) {
          removeCheckpointFile(checkpointQueue.dequeue)
        } else {
          canDelete = false
        }
      }
    }
  }

  /** Checkpoint the Dataset */
  protected def checkpoint(data: RDD[T]): Unit = {
    data.checkpoint()
  }

  /** Return true iff the Dataset is checkpointed */
  protected def isCheckpointed(data: RDD[T]): Boolean = {
    data.isCheckpointed
  }

  /**
    * Persist the Dataset.
    * Note: This should handle checking the current [[StorageLevel]] of the Dataset.
    */
  protected def persist(data: RDD[T]): Unit = {
    if (data.getStorageLevel == StorageLevel.NONE) {
      data.persist(storageLevel)
    }
  }

  /** Unpersist the Dataset */
  protected def unpersist(data: RDD[T]): Unit = {
    data.unpersist(blocking = false)
  }

  /** Call this to unpersist the Dataset. */
  def unpersistDataSet(): Unit = {
    while (persistedQueue.nonEmpty) {
      unpersist(persistedQueue.dequeue)
    }
  }

  /** Call this at the end to delete any remaining checkpoint files. */
  def deleteAllCheckpoints(): Unit = {
    while (checkpointQueue.nonEmpty) {
      removeCheckpointFile(checkpointQueue.dequeue)
    }
  }

  /**
    * Dequeue the oldest checkpointed Dataset, and remove its checkpoint files.
    * This prints a warning but does not fail if the files cannot be removed.
    */
  private def removeCheckpointFile(data: RDD[T]): Unit = {
    // Since the old checkpoint is not deleted by Spark, we manually delete it
    data.getCheckpointFile.foreach { file =>
      Future {
        val start = System.nanoTime
        val path = new Path(file)
        val fs = path.getFileSystem(sc.hadoopConfiguration)
        fs.delete(path, true)
        (System.nanoTime - start) / 1e9

      }.onComplete {
        case Success(v) =>
          logInfo(s"successfully remove old checkpoint file: $file, duration $v seconds")

        case Failure(t) =>
          logWarning(s"fail to remove old checkpoint file: $file, ${t.toString}")
      }
    }
  }
}
