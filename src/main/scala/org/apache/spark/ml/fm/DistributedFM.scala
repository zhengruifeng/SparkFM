package org.apache.spark.ml.fm

import scala.collection.mutable
import scala.reflect._
import scala.{specialized => spec}
import scala.util.Random

import org.apache.hadoop.fs.Path
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._

import org.apache.spark._
import org.apache.spark.ml._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel


class DistributedFM(override val uid: String) extends Estimator[DistributedFMModel]
  with DistributedFMParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("distributed_fm"))

  import DistributedFM._

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setInstanceIndexCol(value: String): this.type = set(instanceIndexCol, value)

  def setFeatureIndiceCol(value: String): this.type = set(featureIndicesCol, value)

  def setFeatureValuesCol(value: String): this.type = set(featureValuesCol, value)

  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)

  def setFitLinear(value: Boolean): this.type = set(fitLinear, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setSeed(value: Long): this.type = set(seed, value)

  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  def setRank(value: Int): this.type = set(rank, value)

  def setNumRandGroups(value: Int): this.type = set(numRandGroups, value)

  def setRegInterceptL1(value: Double): this.type = set(regInterceptL1, value)

  def setRegInterceptL2(value: Double): this.type = set(regInterceptL2, value)

  def setRegLinearL1(value: Double): this.type = set(regLinearL1, value)

  def setRegLinearL2(value: Double): this.type = set(regLinearL2, value)

  def setRegFactorL1(value: Double): this.type = set(regFactorL1, value)

  def setRegFactorL2(value: Double): this.type = set(regFactorL2, value)

  def setRegFactorLG(value: Double): this.type = set(regFactorLG, value)

  def setNumFeatures(value: Long): this.type = set(numFeatures, value)

  def setMaxCCDIters(value: Int): this.type = set(maxCCDIters, value)

  def setInitModelPath(value: String): this.type = set(initModelPath, value)

  def setFloatType(value: String): this.type = set(floatType, value)

  def setIntermediateStorageLevel(value: String): this.type = set(intermediateStorageLevel, value)

  def setFinalStorageLevel(value: String): this.type = set(finalStorageLevel, value)

  override def fit(dataset: Dataset[_]): DistributedFMModel = {

    $(floatType) match {
      case "float" =>
        trainImpl[Float](dataset)

      case "double" =>
        trainImpl[Double](dataset)
    }
  }


  def trainImpl[@spec(Float, Double) V: Fractional : ClassTag](dataset: Dataset[_]): DistributedFMModel = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    require((isDefined(featuresCol) && $(featuresCol).nonEmpty) ||
      (isDefined(featureIndicesCol) && $(featureIndicesCol).nonEmpty
        && isDefined(featureValuesCol) && $(featureValuesCol).nonEmpty))

    val spark = dataset.sparkSession

    val instr = Instrumentation.create(this, dataset)
    instr.logParams(params: _*)

    val numInstances = computeNumInstances(dataset)

    val instances = extractInstances[V](dataset)

    val partitioner = new HashPartitioner(instances.getNumPartitions)

    val flattened = flattenInstances[V](instances, $(numFeatures), partitioner)

    flattened.persist(StorageLevel.fromString($(intermediateStorageLevel)))
    flattened.count()

    val checkpointer = new Checkpointer[(Long, (V, Array[V]))](spark.sparkContext,
      $(checkpointInterval), StorageLevel.fromString($(intermediateStorageLevel)))

    var (intercept, model) = initialize[V](spark, $(numFeatures))

    checkpointer.update(model)
    model.count()

    var iter = 0
    while (iter < $(maxIter)) {
      instr.log(s"iteration $iter")

      if ($(fitIntercept)) {
        instr.log(s"iteration $iter: update intercept")
        val predictions = predict(flattened, intercept, model, $(rank))
        intercept = updateIntercept(predictions, intercept, model,
          $(rank), fromDouble[V]($(regInterceptL1)), fromDouble[V]($(regInterceptL2)))
      }

      if ($(fitLinear)) {
        var group = 0
        while (group < $(numRandGroups)) {
          instr.log(s"iteration $iter: update linear $group")
          val predictions = predict(flattened, intercept, model, $(rank))
          model = updateLinears(predictions, intercept, model, partitioner, $(rank),
            fromDouble[V]($(regLinearL1)), fromDouble[V]($(regLinearL2)),
            $(numRandGroups), group, $(seed) + iter)

          checkpointer.update(model)
          model.count()

          group += 1
        }
      }

      {
        var group = 0
        while (group < $(numRandGroups)) {
          instr.log(s"iteration $iter: update factors $group")
          val predictions = predict(flattened, intercept, model, $(rank))
          model = updateFactors(predictions, intercept, model, partitioner, $(rank),
            fromDouble[V]($(regFactorL1)), fromDouble[V]($(regFactorL2)), fromDouble[V]($(regFactorLG)),
            $(maxCCDIters), $(numRandGroups), group, $(seed) - iter)

          checkpointer.update(model)
          model.count()

          group += 1
        }
      }

      iter += 1
    }

    val finalModel = createFinalModel(spark, model)
    finalModel.persist(StorageLevel.fromString($(finalStorageLevel)))
    finalModel.count()

    checkpointer.unpersistDataSet()
    checkpointer.deleteAllCheckpoints()
    flattened.unpersist(false)

    val fmm = new DistributedFMModel(uid, intercept.toDouble, finalModel)
    instr.logSuccess(fmm)
    copyValues(fmm)
  }


  def computeNumInstances(dataset: Dataset[_]): Long = {
    val spark = dataset.sparkSession
    import spark.implicits._

    val n = $(numFeatures)

    if (isDefined(featuresCol) && $(featuresCol).nonEmpty) {
      dataset.select($(featuresCol))
        .map { row =>
          val vec = row.getAs[Vector](0)
          require(vec.size == n)
          var prevIndex = Int.MinValue
          vec.foreachActive { case (i, v) =>
            require(!v.isNaN)
            require(0 <= i && i < n)
            require(prevIndex <= i)
            prevIndex = i
          }
          true
        }.count()

    } else {
      dataset.select(col($(featureIndicesCol)).cast(ArrayType(LongType)),
        col($(featureValuesCol)).cast(ArrayType(DoubleType)))
        .as[(Array[Long], Array[Double])]
        .map { case (indices, values) =>
          require(indices.length == values.length)
          var prevIndex = Long.MinValue
          indices.iterator
            .zip(values.iterator)
            .foreach { case (i, v) =>
              require(!v.isNaN)
              require(0 <= i && i < n)
              require(prevIndex <= i)
              prevIndex = i
            }
          true
        }.count()
    }
  }


  def extractInstances[@spec(Float, Double) V: Fractional : ClassTag](dataset: Dataset[_]): RDD[(Long, V, V, Array[Long], Array[V])] = {
    val spark = dataset.sparkSession
    import spark.implicits._

    val w = if (isDefined(weightCol) && $(weightCol).nonEmpty) {
      col($(weightCol))
    } else {
      lit(1.0)
    }

    val rdd = if (isDefined(featuresCol) && $(featuresCol).nonEmpty) {
      dataset.select(col($(featuresCol)),
        col($(labelCol)).cast(DoubleType),
        w.cast(DoubleType),
        col($(instanceIndexCol)).cast(LongType))
        .as[(Vector, Double, Double, Long)]
        .mapPartitions { it =>
          val indicesBuilder = mutable.ArrayBuilder.make[Long]
          val valuesBuilder = mutable.ArrayBuilder.make[Double]

          it.map { case (features, instanceLabel, instanceWeight, instanceIndex) =>
            indicesBuilder.clear()
            valuesBuilder.clear()
            features.foreachActive { case (i, v) =>
              if (v != 0) {
                indicesBuilder += i.toLong
                valuesBuilder += v
              }
            }
            (instanceIndex, instanceLabel, instanceWeight, indicesBuilder.result(), valuesBuilder.result())
          }
        }.rdd

    } else {
      dataset.select(
        col($(instanceIndexCol)).cast(LongType),
        col($(labelCol)).cast(DoubleType),
        w.cast(DoubleType),
        col($(featureIndicesCol)).cast(ArrayType(LongType)),
        col($(featureValuesCol)).cast(ArrayType(DoubleType)))
        .as[(Long, Double, Double, Array[Long], Array[Double])]
        .rdd
    }

    rdd.map { case (instanceIndex, instanceLabel, instanceWeight, indices, values) =>
      (instanceIndex, fromDouble[V](instanceLabel), fromDouble[V](instanceWeight), indices, fromDouble[V](values))
    }
  }

  def initialize[@spec(Float, Double) V: Fractional : ClassTag](spark: SparkSession,
                                                                numFeatures: Long): (V, RDD[(Long, (V, Array[V]))]) = {
    import spark.implicits._

    val (intercept, modelDF) = if (isSet(initModelPath) && $(initModelPath).nonEmpty) {
      val m = DistributedFMModel.load($(initModelPath))
      (m.intercept, m.getEntireModel)

    } else {
      val randCols = Array.range(0, $(rank)).map(i => randn($(seed) + i))

      (0.0,
        spark.range(numFeatures).toDF("index")
          .withColumns(Seq("linear", "factor"), Seq(lit(0.0), array(randCols: _*))))
    }

    val modelRDD = modelDF.select(col("index").cast(LongType),
      col("linear").cast(DoubleType),
      col("factor").cast(ArrayType(DoubleType)))
      .as[(Long, Double, Array[Double])]
      .rdd
      .map { case (index, linear, factor) =>
        (index, (fromDouble[V](linear), fromDouble[V](factor)))
      }

    (fromDouble[V](intercept), modelRDD)
  }


  def createFinalModel[@spec(Float, Double) V: Fractional : ClassTag](spark: SparkSession,
                                                                      model: RDD[(Long, (V, Array[V]))]): DataFrame = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    import spark.implicits._

    ($(floatType), $(fitLinear)) match {
      case ("float", true) =>
        model.flatMap { case (index, (linear, factor)) =>
          if (linear != zero || factor.exists(_ != zero)) {
            Iterator.single((index, linear.toFloat, factor.map(_.toFloat)))
          } else {
            Iterator.empty
          }
        }.toDF("index", "linear", "factor")


      case ("double", true) =>
        model.flatMap { case (index, (linear, factor)) =>
          if (linear != zero || factor.exists(_ != zero)) {
            Iterator.single((index, linear.toDouble, factor.map(_.toDouble)))
          } else {
            Iterator.empty
          }
        }.toDF("index", "linear", "factor")

      case ("float", false) =>
        model.flatMap { case (index, (_, factor)) =>
          if (factor.exists(_ != zero)) {
            Iterator.single((index, factor.map(_.toFloat)))
          } else {
            Iterator.empty
          }
        }.toDF("index", "factor")

      case ("double", false) =>
        model.flatMap { case (index, (_, factor)) =>
          if (factor.exists(_ != zero)) {
            Iterator.single((index, factor.map(_.toDouble)))
          } else {
            Iterator.empty
          }
        }.toDF("index", "factor")
    }
  }

  override def copy(extra: ParamMap): DistributedFM = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}


object DistributedFM extends Serializable {

  def fromDouble[@spec(Float, Double) V: Fractional : ClassTag](value: Double): V = {
    classTag[V] match {
      case ClassTag.Float =>
        value.toFloat.asInstanceOf[V]
      case ClassTag.Double =>
        value.asInstanceOf[V]
    }
  }

  def fromDouble[@spec(Float, Double) V: Fractional : ClassTag](array: Array[Double]): Array[V] = {
    classTag[V] match {
      case ClassTag.Float =>
        array.map(_.toFloat).asInstanceOf[Array[V]]
      case ClassTag.Double =>
        array.asInstanceOf[Array[V]]
    }
  }

  def updateIntercept[@spec(Float, Double) V: Fractional : ClassTag](predictions: RDD[(Long, V, V, Array[Long], Array[V], V, Array[V])],
                                                                     intercept: V,
                                                                     model: RDD[(Long, (V, Array[V]))],
                                                                     rank: Int,
                                                                     regI1: V,
                                                                     regI2: V): V = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    val (errorSum, weightSum) = predictions.treeAggregate((zero, zero))(
      seqOp = {
        case ((errorSum, weightSum), (_, instanceLabel, instanceWeight, _, _, prediction, _)) =>
          (errorSum + (instanceLabel - prediction) * instanceWeight, weightSum + instanceWeight)
      }, combOp = {
        case ((errorSum1, weightSum1), (errorSum2, weightSum2)) =>
          (errorSum1 + errorSum2, weightSum1 + weightSum2)
      })

    var a = weightSum + regI2
    a += a

    var c = errorSum + intercept * weightSum
    c += c

    if (c + regI1 < zero) {
      (c + regI1) / a
    } else if (c - regI1 > zero) {
      (c - regI1) / a
    } else {
      zero
    }
  }


  def updateLinears[@spec(Float, Double) V: Fractional : ClassTag](predictions: RDD[(Long, V, V, Array[Long], Array[V], V, Array[V])],
                                                                   intercept: V,
                                                                   model: RDD[(Long, (V, Array[V]))],
                                                                   partitioner: Partitioner,
                                                                   rank: Int,
                                                                   regW1: V,
                                                                   regW2: V,
                                                                   denominator: Int,
                                                                   remainder: Int,
                                                                   seed: Long): RDD[(Long, (V, Array[V]))] = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    val selector = IndicesSelector(denominator, remainder, seed)

    val selectedModel = model.flatMap { case (index, (linear, _)) =>
      if (selector.contains(index)) {
        Iterator.single(index, linear)
      } else {
        Iterator.empty
      }
    }

    val problems = predictions.flatMap { case (_, instanceLabel, instanceWeight, indices, values, prediction, dots) =>
      indices.iterator
        .zip(values.iterator)
        .filter { case (index, _) => selector.contains(index) }
        .map { case (index, value) =>
          (index, (instanceLabel, instanceWeight, value, prediction, dots))
        }

    }.join(selectedModel)

      .map { case (index, ((instanceLabel, instanceWeight, value, prediction, _), linear)) =>
        val y = instanceLabel - prediction + value * linear
        (index, (instanceWeight, y, Array(value), Array.empty[V]))
      }

    val solutions = solve(problems, partitioner, 1, regW1, regW2, zero, 1)

    model.leftOuterJoin(solutions, partitioner)
      .mapPartitions(f = { it =>
        it.map { case (index, ((linear, factor), solution)) =>
          if (solution.nonEmpty) {
            (index, (solution.get.head, factor))
          } else if (selector.contains(index)) {
            (index, (zero, factor))
          } else {
            (index, (linear, factor))
          }
        }
      }, true)
  }


  def updateFactors[@spec(Float, Double) V: Fractional : ClassTag](predictions: RDD[(Long, V, V, Array[Long], Array[V], V, Array[V])],
                                                                   intercept: V,
                                                                   model: RDD[(Long, (V, Array[V]))],
                                                                   partitioner: Partitioner,
                                                                   rank: Int,
                                                                   regV1: V,
                                                                   regV2: V,
                                                                   regVG: V,
                                                                   ccdIters: Int,
                                                                   denominator: Int,
                                                                   remainder: Int,
                                                                   seed: Long): RDD[(Long, (V, Array[V]))] = {

    val fracV = implicitly[Fractional[V]]
    import fracV._

    val selector = IndicesSelector(denominator, remainder, seed)

    val selectedModel = model.flatMap { case (index, (_, factor)) =>
      if (selector.contains(index)) {
        Iterator.single(index, factor)
      } else {
        Iterator.empty
      }
    }

    val problems = predictions.flatMap { case (_, instanceLabel, instanceWeight, indices, values, prediction, dots) =>
      indices.iterator
        .zip(values.iterator)
        .filter { case (index, _) => selector.contains(index) }
        .map { case (index, value) =>
          (index, (instanceLabel, instanceWeight, value, prediction, dots))
        }

    }.join(selectedModel)

      .map { case (index, ((instanceLabel, instanceWeight, value, prediction, dots), factor)) =>
        var y = instanceLabel - prediction
        val x = Array.fill(rank)(zero)

        if (factor.nonEmpty) {
          for (f <- 0 until rank) {
            val vfl = factor(f)
            val r = value * (dots(f) - vfl * value)
            x(f) = r
            y += vfl * r
          }
        } else {
          for (f <- 0 until rank) {
            x(f) = value * dots(f)
          }
        }

        (index, (instanceWeight, y, x, factor))
      }

    val solutions = solve(problems, partitioner, rank, regV1, regV2, regVG, ccdIters)

    model.leftOuterJoin(solutions, partitioner)
      .mapPartitions(f = { it =>
        it.map { case (index, ((linear, factor), solution)) =>
          if (solution.nonEmpty) {
            (index, (linear, solution.get))
          } else if (selector.contains(index)) {
            // two reasons can lead to null solution of feature #i
            // 1: all values in feature #i are zero in dataset flattened, then no problem of feature #i is generated
            // 2: new solution of feature #i is a zero-vector, the solver will ignored zero-vectors
            (index, (linear, Array.empty[V]))
          } else {
            (index, (linear, factor))
          }
        }
      }, true)
  }


  /**
    * input rdd:
    * instances
    * _1           instance_index   Long
    * _2           instance_label   V
    * _3           instance_weight  V
    * _4           indices          Array[Long]     non-zero indices
    * _5           values           Array[V]
    *
    */
  def flattenInstances[@spec(Float, Double) V: Fractional : ClassTag](instances: RDD[(Long, V, V, Array[Long], Array[V])],
                                                                      numFeatures: Long,
                                                                      partitioner: Partitioner): RDD[(Long, (Long, V, V, V, Array[Long], Array[V]))] = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    instances.mapPartitionsWithIndex { case (pid, it) =>
      val rng = new Random(pid)

      it.flatMap { case (instanceIndex, instanceLabel, instanceWeight, indices, values) =>
        if (indices.nonEmpty) {
          var first = true
          indices.iterator
            .zip(values.iterator)
            .map { case (index, value) =>
              if (first) {
                first = false
                (index, (instanceIndex, instanceLabel, instanceWeight, value, indices, values))
              } else {
                (index, (instanceIndex, instanceLabel, instanceWeight, value, Array.emptyLongArray, Array.empty[V]))
              }
            }

        } else {
          // make sure that each instanceIndex appears in the flattened dataset
          val index = (rng.nextDouble * numFeatures).toLong
          Iterator.single((index, (instanceIndex, instanceLabel, instanceWeight, zero, indices, values)))
        }
      }
    }.partitionBy(partitioner)
  }


  /**
    * input rdds:
    * flattened
    * _1           index            Long
    * _2._1        instance_index   Long
    * _2._2        instance_label   V
    * _2._3        instance_weight  V
    * _2._4        value            V               xl
    * _2._5        indices          Array[Long]     non-zero indices
    * _2._6        values           Array[V]
    *
    * model
    * _1           index            Long            feature index
    * _2._1        linear           V
    * _2._2        factor           Array[V]
    *
    */
  def predict[@spec(Float, Double) V: Fractional : ClassTag](flattened: RDD[(Long, (Long, V, V, V, Array[Long], Array[V]))],
                                                             intercept: V,
                                                             model: RDD[(Long, (V, Array[V]))],
                                                             rank: Int): RDD[(Long, V, V, Array[Long], Array[V], V, Array[V])] = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    val negativeInfinity = -one / zero

    flattened.join(model)

      .map { case (_, ((instanceIndex, instanceLabel, instanceWeight, value, indices, values), (linear, factor))) =>
        (instanceIndex, (instanceLabel, instanceWeight, value, indices, values, linear, factor))

      }.aggregateByKey((new PredictWithDotsAggregator[V](rank), negativeInfinity, negativeInfinity, Array.emptyLongArray, Array.empty[V]))(
      seqOp = {
        case ((predAgg, instanceLabel_, instanceWeight_, indices_, values_),
        (instanceLabel, instanceWeight, value, indices, values, linear, factor)) =>
          (predAgg.update(value, linear, factor),
            max(instanceLabel, instanceLabel_),
            max(instanceWeight, instanceWeight_),
            Seq(indices, indices_).maxBy(_.length),
            Seq(values, values_).maxBy(_.length))

      }, combOp = {
        case ((predAgg1, instanceLabel1, instanceWeight1, indices1, values1),
        (predAgg2, instanceLabel2, instanceWeight2, indices2, values2)) =>
          (predAgg1.merge(predAgg2),
            max(instanceLabel1, instanceLabel2),
            max(instanceWeight1, instanceWeight2),
            Seq(indices1, indices2).maxBy(_.length),
            Seq(values1, values2).maxBy(_.length))

      }).map { case (instanceIndex, (predAgg, instanceLabel, instanceWeight, indices, values)) =>
      val (prediction, dots) = predAgg.compute(intercept)
      (instanceIndex, instanceLabel, instanceWeight, indices, values, prediction, dots)
    }
  }


  /**
    * problems
    * _1           index              Long
    * _2._1        instance_weight    V
    * _2._2        problem_y          V
    * _2._3        problem_x          Array[V]       [x0, x1, ...]
    * _2._4        previous_solution  Array[V]       empty value mines zero-vector
    *
    * Note: if the new solution is a zero-vector, it will be discarded in the output
    */
  def solve[@spec(Float, Double) V: Fractional : ClassTag](problems: RDD[(Long, (V, V, Array[V], Array[V]))],
                                                           partitioner: Partitioner,
                                                           k: Int,
                                                           regL1: V,
                                                           regL2: V,
                                                           regLG: V,
                                                           iters: Int): RDD[(Long, Array[V])] = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    problems.aggregateByKey((new SolverAggregator[V](k), Array.empty[V]), partitioner)(
      seqOp = {
        case ((solverAgg, prevSolution_), (instanceWeight, y, x, prevSolution)) =>
          (solverAgg.update(instanceWeight, x, y),
            Seq(prevSolution, prevSolution_).maxBy(_.length))
      },
      combOp = {
        case ((solverAgg1, prevSolution1), (solverAgg2, prevSolution2)) =>
          (solverAgg1.merge(solverAgg2),
            Seq(prevSolution1, prevSolution2).maxBy(_.length))

      }).mapPartitions(f = { it =>
      it.flatMap { case (index, (solverAgg, prevSolution)) =>
        val prev = if (prevSolution.isEmpty) {
          Array.fill[V](k)(zero)
        } else {
          prevSolution
        }
        val solution = solverAgg.solve(regL1, regL2, regLG, prev, iters)
        if (solution.exists(_ != zero)) {
          Iterator.single((index, solution))
        } else {
          Iterator.empty
        }
      }
    }, true)
  }
}


class PredictWithDotsAggregator[@spec(Float, Double) V: Fractional : ClassTag](val rank: Int) extends Serializable {
  require(rank > 0)

  val fracV = implicitly[Fractional[V]]

  import fracV._

  // first #rank elements store dotSum, the following #rank elements store dot2Sum, the last one stores linearSum
  val stat = Array.fill(rank + rank + 1)(zero)

  def update(value: V, linear: V, factor: Array[V]): PredictWithDotsAggregator[V] = {
    if (value != zero) {
      if (linear != zero) {
        stat(stat.length - 1) += value * linear
      }

      if (factor.nonEmpty) {
        require(factor.length == rank)
        var i = 0
        while (i < rank) {
          val s = value * factor(i)
          stat(i) += s
          stat(rank + i) += s * s
          i += 1
        }
      }
    }
    this
  }

  def merge(o: PredictWithDotsAggregator[V]): PredictWithDotsAggregator[V] = {
    var i = 0
    while (i < stat.length) {
      stat(i) += o.stat(i)
      i += 1
    }
    this
  }

  def compute(intercept: V): (V, Array[V]) = {
    var pred = intercept + stat.last
    var i = 0
    while (i < rank) {
      pred += (stat(i) * stat(i) - stat(rank + i)) / fromInt(2)
      i += 1
    }
    (pred, stat.take(rank))
  }
}


class DistributedFMModel private[ml](override val uid: String,
                                     val intercept: Double,
                                     @transient val model: DataFrame)
  extends Model[DistributedFMModel] with DistributedFMParams with MLWritable {

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setInstanceIndexCol(value: String): this.type = set(instanceIndexCol, value)

  def setFeatureIndiceCol(value: String): this.type = set(featureIndicesCol, value)

  def setFeatureValuesCol(value: String): this.type = set(featureValuesCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val predictUDAF = $(floatType) match {
      case "float" =>
        new FloatPredictUDAF($(rank), intercept.toFloat)
      case "double" =>
        new DoublePredictUDAF($(rank), intercept)
    }

    val linearCol = if ($(fitLinear)) {
      col("linear")
    } else if ($(floatType) == "float") {
      lit(0.0F)
    } else {
      lit(0.0)
    }

    val modelDF = model.select(col("index"),
      linearCol.as("linear"),
      col("factor"))

    val flattened = flattenDataset(dataset)

    flattened.join(modelDF.hint("broadcast"), Seq("index"), "leftouter")
      .groupBy($(instanceIndexCol))
      .agg(predictUDAF(col("value"),
        col("linear"),
        col("factor")).as($(predictionCol)))
      .join(dataset, $(instanceIndexCol))
  }

  def flattenDataset(dataset: Dataset[_]): DataFrame = {
    val spark = model.sparkSession
    import spark.implicits._

    val valueCol = if ($(floatType) == "float") {
      col("value").cast(FloatType)
    } else {
      col("value").cast(DoubleType)
    }

    val instances = if (isDefined(featuresCol) && $(featuresCol).nonEmpty) {
      dataset.select(col($(featuresCol)),
        col($(instanceIndexCol)).cast(LongType))
        .as[(Vector, Long)]
        .mapPartitions { it =>
          val indicesBuilder = mutable.ArrayBuilder.make[Long]
          val valuesBuilder = mutable.ArrayBuilder.make[Double]

          it.map { case (features, instanceIndex) =>
            indicesBuilder.clear()
            valuesBuilder.clear()
            features.foreachActive { case (i, v) =>
              if (v != 0) {
                indicesBuilder += i.toLong
                valuesBuilder += v
              }
            }
            (instanceIndex, indicesBuilder.result(), valuesBuilder.result())
          }
        }

    } else {
      dataset.select(
        col($(instanceIndexCol)).cast(LongType),
        col($(featureIndicesCol)).cast(ArrayType(LongType)),
        col($(featureValuesCol)).cast(ArrayType(DoubleType)))
        .as[(Long, Array[Long], Array[Double])]
    }

    instances.flatMap { case (instanceIndex, indices, values) =>
      require(indices.length == values.length)
      if (indices.nonEmpty) {
        indices.iterator
          .zip(values.iterator)
          .map { case (index, value) =>
            (instanceIndex, index, value)
          }
      } else {
        // make sure that each instanceIndex appears in the flattened dataset
        Iterator.single((instanceIndex, 0L, 0.0))
      }
    }.toDF($(instanceIndexCol), "index", "value")
      .withColumn("value", valueCol)
  }

  override def copy(extra: ParamMap): DistributedFMModel = {
    val copied = new DistributedFMModel(uid, intercept, model)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new DistributedFMModel.DistributedFMModelWriter(this)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  def getEntireModel: DataFrame = {
    val spark = model.sparkSession

    val linearCol = if ($(fitLinear)) {
      col("linear")
    } else if ($(floatType) == "float") {
      lit(0.0F)
    } else {
      lit(0.0)
    }

    val (zeroLinearCol, zeroFactorCol) = $(floatType) match {
      case "float" =>
        (lit(0.0F), lit(Array.fill($(rank))(0.0F)))
      case "double" =>
        (lit(0.0), lit(Array.fill($(rank))(0.0)))
    }

    spark.range($(numFeatures)).toDF("index")
      .join(model.withColumn("linear", linearCol), Seq("index"), "outer")
      .select(col("index"),
        when(col("linear").isNull, zeroLinearCol)
          .otherwise(col("linear")).as("linear"),
        when(col("factor").isNull, zeroFactorCol)
          .otherwise(col("factor")).as("factor"))
  }
}


object DistributedFMModel extends MLReadable[DistributedFMModel] {

  override def read: MLReader[DistributedFMModel] = new DistributedFMModelReader

  override def load(path: String): DistributedFMModel = super.load(path)

  private[DistributedFMModel] class DistributedFMModelWriter(instance: DistributedFMModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      val extraMetadata = "intercept" -> instance.intercept
      DefaultParamsWriter.saveMetadata(instance, path, sc, Some(extraMetadata))
      val modelPath = new Path(path, "model").toString
      instance.model.write.parquet(modelPath)
    }
  }

  private class DistributedFMModelReader extends MLReader[DistributedFMModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[DistributedFMModel].getName

    override def load(path: String): DistributedFMModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      implicit val format = DefaultFormats
      val intercept = (metadata.metadata \ "intercept").extract[Double]

      val modelPath = new Path(path, "model").toString
      val df = sparkSession.read.parquet(modelPath)
      val model = new DistributedFMModel(metadata.uid, intercept, df)

      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

}


class FloatPredictUDAF(val rank: Int,
                       val intercept: Float) extends UserDefinedAggregateFunction {

  override def inputSchema: StructType = StructType(
    StructField("value", FloatType) ::
      StructField("linear", FloatType) ::
      StructField("factor", ArrayType(FloatType)) :: Nil)

  override def bufferSchema: StructType = StructType(
    StructField("stat", ArrayType(FloatType)) :: Nil)

  override def dataType: DataType = FloatType

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    // first #rank elements store dotSum, the following #rank elements store dot2Sum, the last one stores linearSum
    buffer(0) = Array.ofDim[Float](rank + rank + 1)
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val v = input.getFloat(0)

    if (v != 0) {
      val stat = buffer.getSeq[Float](0).toArray

      if (!input.isNullAt(1)) {
        val linear = input.getFloat(1)
        if (linear != 0) {
          stat(stat.length - 1) += v * linear
        }
      }

      if (!input.isNullAt(2)) {
        val factor = input.getSeq[Float](2).toArray
        if (factor.nonEmpty) {
          require(factor.length == rank)
          var i = 0
          while (i < rank) {
            val s = v * factor(i)
            stat(i) += s
            stat(rank + i) += s * s
            i += 1
          }
        }
      }

      buffer(0) = stat
    }
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    val stat1 = buffer1.getSeq[Float](0).toArray
    val stat2 = buffer2.getSeq[Float](0).toArray
    require(stat1.length == stat2.length)
    var i = 0
    while (i < stat1.length) {
      stat1(i) += stat2(i)
      i += 1
    }
    buffer1(0) = stat1
  }

  override def evaluate(buffer: Row): Any = {
    val stat = buffer.getSeq[Float](0).toArray
    var prediction = intercept + stat.last
    var i = 0
    while (i < rank) {
      prediction += (stat(i) * stat(i) - stat(rank + i)) / 2
      i += 1
    }
    prediction
  }
}


class DoublePredictUDAF(val rank: Int,
                        val intercept: Double) extends UserDefinedAggregateFunction {

  override def inputSchema: StructType = StructType(
    StructField("value", DoubleType) ::
      StructField("linear", DoubleType) ::
      StructField("factor", ArrayType(DoubleType)) :: Nil)

  override def bufferSchema: StructType = StructType(
    StructField("stat", ArrayType(DoubleType)) :: Nil)

  override def dataType: DataType = DoubleType

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = Array.ofDim[Double](rank + rank + 1)
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val v = input.getDouble(0)

    if (v != 0) {
      val stat = buffer.getSeq[Double](0).toArray

      if (!input.isNullAt(1)) {
        val linear = input.getDouble(1)
        if (linear != 0) {
          stat(stat.length - 1) += v * linear
        }
      }

      if (!input.isNullAt(2)) {
        val factor = input.getSeq[Double](2).toArray
        if (factor.nonEmpty) {
          require(factor.length == rank)
          var i = 0
          while (i < rank) {
            val s = v * factor(i)
            stat(i) += s
            stat(rank + i) += s * s
            i += 1
          }
        }
      }

      buffer(0) = stat
    }
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    val stat1 = buffer1.getSeq[Double](0).toArray
    val stat2 = buffer2.getSeq[Double](0).toArray
    require(stat1.length == stat2.length)
    var i = 0
    while (i < stat1.length) {
      stat1(i) += stat2(i)
      i += 1
    }
    buffer1(0) = stat1
  }

  override def evaluate(buffer: Row): Any = {
    val stat = buffer.getSeq[Double](0).toArray
    var prediction = intercept + stat.last
    var i = 0
    while (i < rank) {
      prediction += (stat(i) * stat(i) - stat(rank + i)) / 2
      i += 1
    }
    prediction
  }
}