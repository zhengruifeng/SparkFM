package org.apache.spark.ml.fm

import scala.collection.mutable
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.util.{Failure, Success}

import org.apache.hadoop.fs.Path
import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.unsafe.hash.Murmur3_x86_32


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


case class IndicesSelector(denominator: Int,
                           remainder: Int,
                           seed: Long) {
  require(denominator > 0)
  require(remainder >= 0 && remainder < denominator)

  def contains(index: Long): Boolean = {
    Murmur3_x86_32.hashLong(index, seed.toInt).abs % denominator == remainder
  }
}

object Utils extends Serializable {


  def cleanShuffleDependencies(sc: SparkContext,
                               deps: Seq[Dependency[_]],
                               blocking: Boolean = false): Unit = {
    // If there is no reference tracking we skip clean up.
    sc.cleaner.foreach { cleaner =>

      /**
        * Clean the shuffles & all of its parents.
        */
      def cleanEagerly(dep: Dependency[_]): Unit = {
        if (dep.isInstanceOf[ShuffleDependency[_, _, _]]) {
          val shuffleId = dep.asInstanceOf[ShuffleDependency[_, _, _]].shuffleId
          cleaner.doCleanupShuffle(shuffleId, blocking)
        }
        val rdd = dep.rdd
        val rddDeps = rdd.dependencies
        if (rdd.getStorageLevel == StorageLevel.NONE && rddDeps != null) {
          rddDeps.foreach(cleanEagerly)
        }
      }

      deps.foreach(cleanEagerly)
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

      Utils.cleanShuffleDependencies(sc, data.dependencies)
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


private[fm] class DataFrameCheckpointer(val spark: SparkSession,
                                        val checkpointInterval: Int,
                                        val checkpointDir: String,
                                        val storageLevel: StorageLevel,
                                        val maxPersisted: Int) extends Logging {
  def this(spark: SparkSession, checkpointInterval: Int, checkpointDir: String, storageLevel: StorageLevel) =
    this(spark, checkpointInterval, checkpointDir, storageLevel, 2)

  require(storageLevel != StorageLevel.NONE)
  require(maxPersisted > 1)

  /** FIFO queue of past checkpointed Datasets */
  private val checkpointedFileQueue = mutable.Queue.empty[String]

  /** FIFO queue of past persisted Datasets */
  private val persistedQueue = mutable.Queue.empty[DataFrame]

  /** Number of times [[update()]] has been called */
  private var updateCount = 0

  /**
    * Update with a new Dataset. Handle persistence and checkpointing as needed.
    * Since this handles persistence and checkpointing, this should be called before the Dataset
    * has been materialized.
    *
    * @param data New Dataset created from previous Datasets in the lineage.
    */
  def update(data: DataFrame): DataFrame = {
    val newData = if (checkpointDir.nonEmpty &&
      checkpointInterval != -1 && (updateCount % checkpointInterval) == 0) {
      val file = s"${checkpointDir}/DataFrame_Snapshot_${updateCount}"
      data.write.mode(SaveMode.Overwrite).parquet(file)
      checkpointedFileQueue.enqueue(file)
      spark.read.parquet(file)
    } else {
      truncate(data)
    }

    newData.persist(storageLevel)
    persistedQueue.enqueue(newData)

    while (checkpointedFileQueue.length > 1) {
      removeCheckpointFile(checkpointedFileQueue.dequeue)
    }

    while (persistedQueue.length > maxPersisted) {
      persistedQueue.dequeue.unpersist(false)
    }

    updateCount += 1

    newData
  }

  def cleanup(): Unit = {
    while (checkpointedFileQueue.nonEmpty) {
      removeCheckpointFile(checkpointedFileQueue.dequeue)
    }

    while (persistedQueue.nonEmpty) {
      persistedQueue.dequeue.unpersist(false)
    }
  }

  def truncate(df: DataFrame): DataFrame = {
    val rdd = df.rdd
    val schema = df.schema
    spark.createDataFrame(rdd, schema)
  }

  private def removeCheckpointFile(file: String): Unit = {
    Future {
      val start = System.nanoTime
      val path = new Path(file)
      val fs = path.getFileSystem(spark.sparkContext.hadoopConfiguration)
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