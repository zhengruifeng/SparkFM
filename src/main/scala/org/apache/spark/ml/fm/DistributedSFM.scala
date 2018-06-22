package org.apache.spark.ml.fm

import org.apache.hadoop.fs.Path

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.{specialized => spec}

import org.apache.spark.ml._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel

class DistributedSFM(override val uid: String) extends Estimator[DistributedSFMModel]
  with DistributedFMParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("distributed_sfm"))

  import DistributedSFM._

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

  def setCheckpointDir(value: String): this.type = set(checkpointDir, value)

  def setRank(value: Int): this.type = set(rank, value)

  def setNumRandGroups(value: Int): this.type = set(numRandGroups, value)

  def setRegInterceptL1(value: Double): this.type = set(regInterceptL1, value)

  def setRegInterceptL2(value: Double): this.type = set(regInterceptL2, value)

  def setRegLinearL1(value: Double): this.type = set(regLinearL1, value)

  def setRegLinearL2(value: Double): this.type = set(regLinearL2, value)

  def setRegFactorL1(value: Double): this.type = set(regFactorL1, value)

  def setRegFactorL2(value: Double): this.type = set(regFactorL2, value)

  def setRegFactorLG(value: Double): this.type = set(regFactorLG, value)

  def setNumFeaturesG(value: Long): this.type = set(numFeatures, value)

  def setMaxCCDIters(value: Int): this.type = set(maxCCDIters, value)

  def setInitModelPath(value: String): this.type = set(initModelPath, value)

  def setIntermediateStorageLevel(value: String): this.type = set(intermediateStorageLevel, value)

  def setFinalStorageLevel(value: String): this.type = set(finalStorageLevel, value)

  override def fit(dataset: Dataset[_]): DistributedSFMModel = {

    require((isDefined(featuresCol) && $(featuresCol).nonEmpty) ||
      (isDefined(featureIndicesCol) && $(featureIndicesCol).nonEmpty
        && isDefined(featureValuesCol) && $(featureValuesCol).nonEmpty))

    val spark = dataset.sparkSession
    import spark.implicits._

    val maxFeature = computeMaxFeature(dataset)

    val numFeatures = maxFeature

    val instances = extractInstances(dataset)

    val handlePersistence = dataset.storageLevel == StorageLevel.NONE
    if (handlePersistence) {
      instances.persist(StorageLevel.fromString($(intermediateStorageLevel)))
    }
    instances.count()


    //    val storageLevel = StorageLevel.fromString($(intermediateStorageLevel))


    val checkpointer = new Checkpointer[(Long, (Float, Array[Float]))](spark.sparkContext,
      $(checkpointInterval), StorageLevel.fromString($(intermediateStorageLevel)))

    var (intercept, modelDF) = initialize(spark, numFeatures)
    modelDF.show(10, false)

    var model: RDD[(Long, (Float, Array[Float]))] = modelDF.select("index", "linear", "factor")
      .as[(Long, Float, Array[Float])]
      .rdd
      .map { case (index, linear, factor) =>
        (index, (linear, factor))
      }

    //    model.persist(storageLevel)
    //    model.count()

    checkpointer.update(model)
    model.count()

    val instr = Instrumentation.create(this, dataset)
    instr.logParams(params: _*)

    var iter = 0
    while (iter < $(maxIter)) {
      instr.log(s"iteration $iter")

      if ($(fitIntercept)) {
        instr.log(s"iteration $iter: update intercept")
        intercept = updateIntercept[Float](instances, intercept, model,
          $(rank), $(regInterceptL1).toFloat, $(regInterceptL2).toFloat)

        Seq(intercept).toDF("intercept").show
      }

      if ($(fitLinear)) {
        var group = 0
        while (group < $(numRandGroups)) {
          instr.log(s"iteration $iter: update linear $group")

          //          val prevModel = model

          val (newModel, modified) = updateLinears[Float](instances, intercept, model,
            $(rank), $(regLinearL1).toFloat, $(regLinearL2).toFloat,
            $(numRandGroups), group, $(seed) + iter)

          if (modified) {
            model = newModel
//            checkpointer.update(model)
//            //            model.persist(storageLevel)
//            model.count()

            val path = s"${$(checkpointDir)}/$iter-linear-$group"
            model.saveAsObjectFile(path)

            model = spark.sparkContext.objectFile[(Long, (Float, Array[Float]))](path)

            //            prevModel.unpersist(false)

          } else {
            Seq("non-modification").toDF("linear").show(10, false)
          }

          model.map {
            case (index, (linear, factor)) =>
              (index, linear, factor)
          }.toDF("index", "linear", "factor").sort("index").show(10, false)

          group += 1
        }
      }

      {
        var group = 0
        while (group < $(numRandGroups)) {
          instr.log(s"iteration $iter: update factors $group")

          //          val prevModel = model

          val (newModel, modified) = updateFactors(instances, intercept, model,
            $(rank), $(regFactorL1).toFloat, $(regFactorL2).toFloat, $(regFactorLG).toFloat,
            $(maxCCDIters), $(numRandGroups), group, $(seed) - iter)

          if (modified) {
            model = newModel
//            checkpointer.update(model)
            //            model.persist(storageLevel)
//            model.count()

            val path = s"${$(checkpointDir)}/$iter-factors-$group"
            model.saveAsObjectFile(path)

            model = spark.sparkContext.objectFile[(Long, (Float, Array[Float]))](path)

            //            prevModel.unpersist(false)

          } else {
            Seq("non-modification").toDF("factor").show(10, false)
          }

          model.map {
            case (index, (linear, factor)) =>
              (index, linear, factor)
          }.toDF("index", "linear", "factor").sort("index").show(10, false)

          group += 1
        }
      }

      iter += 1
    }

    val finalModel = model.map {
      case (index, (linear, factor)) =>
        (index, linear, factor)
    }.toDF("index", "linear", "factor")

    finalModel.persist(StorageLevel.fromString($(finalStorageLevel)))
    finalModel.count()

    //    checkpointer.unpersistDataSet()
    //    checkpointer.deleteAllCheckpoints()

    if (handlePersistence) {
      instances.unpersist(false)
    }

    val fmm = new DistributedSFMModel(uid, intercept, finalModel)
    instr.logSuccess(fmm)
    copyValues(fmm)
  }

  def computeMaxFeature(dataset: Dataset[_]): Long = {
    val spark = dataset.sparkSession
    import spark.implicits._

    if (isDefined(featuresCol) && $(featuresCol).nonEmpty) {
      dataset.select($(featuresCol)).head()
        .getAs[Vector](0).size.toLong

    } else {
      dataset.select($(featureIndicesCol))
        .as[Array[Long]].rdd
        .map { indices =>
          if (indices.nonEmpty) {
            indices.max
          } else {
            0L
          }
        }.max() + 1
    }
  }

  def extractInstances(dataset: Dataset[_]): RDD[(Long, Float, Float, Array[Long], Array[Float])] = {
    val spark = dataset.sparkSession
    import spark.implicits._

    val w = if (isDefined(weightCol) && $(weightCol).nonEmpty) {
      col($(weightCol))
    } else {
      lit(1.0F)
    }

    if (isDefined(featuresCol) && $(featuresCol).nonEmpty) {
      dataset.select(col($(featuresCol)),
        col($(labelCol)).cast(FloatType),
        w.cast(FloatType),
        col($(instanceIndexCol)).cast(LongType))
        .as[(Vector, Float, Float, Long)]
        .mapPartitions { it =>
          val indicesBuilder = mutable.ArrayBuilder.make[Long]
          val valuesBuilder = mutable.ArrayBuilder.make[Float]

          it.map { case (features, instanceLabel, instanceWeight, instanceIndex) =>
            indicesBuilder.clear()
            valuesBuilder.clear()
            features.foreachActive { case (i, v) =>
              if (v != 0) {
                indicesBuilder += i.toLong
                valuesBuilder += v.toFloat
              }
            }
            (instanceIndex, instanceLabel, instanceWeight, indicesBuilder.result(), valuesBuilder.result())
          }
        }.rdd

    } else {
      dataset.select(
        col($(instanceIndexCol)).cast(LongType),
        col($(labelCol)).cast(FloatType),
        w.cast(FloatType),
        col($(featureIndicesCol)).cast(ArrayType(LongType)),
        col($(featureValuesCol)).cast(ArrayType(FloatType)))
        .as[(Long, Float, Float, Array[Long], Array[Float])]
        .rdd
    }
  }

  def initialize(spark: SparkSession,
                 numFeatures: Long): (Float, DataFrame) = {
    if (isSet(initModelPath) && $(initModelPath).nonEmpty) {
      val m = DistributedFMModel.load($(initModelPath))
      (m.intercept, m.model)

    } else {
      val randCols = Array.range(0, $(rank))
        .map(i => randn($(seed) + i).cast(FloatType))

      (0.0F,
        spark.range(numFeatures).toDF("index")
          .withColumns(Seq("linear", "factor"), Seq(lit(0.0F), array(randCols: _*))))
    }
  }


  override def copy(extra: ParamMap): DistributedSFM = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}


object DistributedSFM extends Serializable {


  /**
    * input
    * 00           instance_index   Long
    * 01           instance_label   Float
    * 02           instance_weight  Float
    * 03           indices          Array[Long]     non-zero indices
    * 04           values           Array[Float]
    *
    * model
    * 00           index            Long            feature index
    * 01           linear           Float
    * 02           factor           Array[Float]
    */
  def updateIntercept[@spec(Float, Double) V: Fractional : ClassTag](instances: RDD[(Long, V, V, Array[Long], Array[V])],
                                                                     intercept: V,
                                                                     model: RDD[(Long, (V, Array[V]))],
                                                                     rank: Int,
                                                                     regI1: V,
                                                                     regI2: V): V = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    val (errorSum, weightSum) =
      instances.flatMap { case (instanceIndex, instanceLabel, instanceWeight, indices, values) =>
        indices.iterator
          .zip(values.iterator)
          .filter(t => t._2 != zero)
          .map { case (index, value) =>
            (index, (instanceIndex, instanceLabel, instanceWeight, value))
          }

      }.join(model)

        .map { case (index, ((instanceIndex, instanceLabel, instanceWeight, value), (linear, factor))) =>
          (instanceIndex, (instanceLabel, instanceWeight, value, linear, factor))

        }.aggregateByKey((new PredictWithDotsAggregator[V](rank), zero, zero))(
        seqOp = {
          case ((predAgg, _, _), (instanceLabel, instanceWeight, value, linear, factor)) =>
            (predAgg.update(value, linear, factor), instanceLabel, instanceWeight)
        }, combOp = {
          case ((predAgg1, instanceLabel1, instanceWeight1), (predAgg2, _, _)) =>
            (predAgg1.merge(predAgg2), instanceLabel1, instanceWeight1)

        }).map { case (_, (predAgg, instanceLabel, instanceWeight)) =>
        val (prediction, _) = predAgg.compute(intercept)
        ((instanceLabel - prediction) * instanceWeight, instanceWeight)

      }.treeReduce(f = {
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


  /**
    * input
    * 00           instance_index   Long
    * 01           instance_label   Float
    * 02           instance_weight  Float
    * 03           indices          Array[Long]     non-zero indices
    * 04           values           Array[Float]
    *
    * model
    * 00           index            Long            feature index
    * 01           linear           Float
    * 02           factor           Array[Float]
    * 03           selected         Bool
    */
  def updateLinears[@spec(Float, Double) V: Fractional : ClassTag](instances: RDD[(Long, V, V, Array[Long], Array[V])],
                                                                   intercept: V,
                                                                   model: RDD[(Long, (V, Array[V]))],
                                                                   rank: Int,
                                                                   regW1: V,
                                                                   regW2: V,
                                                                   denominator: Int,
                                                                   remainder: Int,
                                                                   seed: Long): (RDD[(Long, (V, Array[V]))], Boolean) = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    val selector = IndicesSelector(denominator, remainder, seed)
    val selectedModel = model.flatMap { case (index, (linear, _)) =>
      if (selector.contain(index)) {
        Iterator.single(index, linear)
      } else {
        Iterator.empty
      }
    }

    if (selectedModel.isEmpty) {
      (model, false)

    } else {
      val predicted = predictAndFlatten(instances, intercept, model, rank, selector)

      val problems = predicted.join(selectedModel)
        .map { case (index, ((instanceLabel, instanceWeight, value, prediction, _), linear)) =>
          val y = instanceLabel - prediction + value * linear
          (index, (instanceWeight, y, Array(value), Array(zero)))
        }

      val solutions = solve(problems, 1, regW1, regW2, zero, 1)

      val newModel = model.leftOuterJoin(solutions)
        .map { case (index, ((linear, factor), solution)) =>
          if (solution.nonEmpty) {
            (index, (solution.get.head, factor))
          } else {
            (index, (linear, factor))
          }
        }

      (newModel, true)
    }
  }


  /**
    * input
    * 00           instance_index   Long
    * 01           instance_label   Float
    * 02           instance_weight  Float
    * 03           indices          Array[Long]     non-zero indices
    * 04           values           Array[Float]
    *
    * model
    * 00           index            Long            feature index
    * 01           linear           Float
    * 02           factor           Array[Float]
    * 03           selected         Bool
    */
  def updateFactors[@spec(Float, Double) V: Fractional : ClassTag](instances: RDD[(Long, V, V, Array[Long], Array[V])],
                                                                   intercept: V,
                                                                   model: RDD[(Long, (V, Array[V]))],
                                                                   rank: Int,
                                                                   regV1: V,
                                                                   regV2: V,
                                                                   regVG: V,
                                                                   ccdIters: Int,
                                                                   denominator: Int,
                                                                   remainder: Int,
                                                                   seed: Long): (RDD[(Long, (V, Array[V]))], Boolean) = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    val selector = IndicesSelector(denominator, remainder, seed)
    val selectedModel = model.flatMap { case (index, (_, factor)) =>
      if (selector.contain(index)) {
        Iterator.single(index, factor)
      } else {
        Iterator.empty
      }
    }

    if (selectedModel.isEmpty) {
      (model, false)

    } else {

      val spark = SparkSession.builder().getOrCreate()
      import spark.implicits._

      val predicted = predictAndFlatten(instances, intercept, model, rank, selector)

      predicted.map { case (index, (instanceLabel, instanceWeight, value, prediction, dots)) =>
        (index, instanceLabel.toFloat(), instanceWeight.toFloat(), value.toFloat(), prediction.toFloat(), dots.map(_.toFloat()))
      }.toDF("index", "label", "weight", "value", "prediction", "dots")
        .sort("index")
        .show(10, false)

      val problems = predicted.join(selectedModel)
        .map { case (index, ((instanceLabel, instanceWeight, value, prediction, dots), factor)) =>
          var y = instanceLabel - prediction
          val x = Array.fill(rank)(zero)
          for (f <- 0 until rank) {
            val vfl = factor(f)
            val r = value * (dots(f) - vfl * value)
            x(f) = r
            y += vfl * r
          }
          (index, (instanceWeight, y, x, factor))
        }


      val solutions = solve(problems, rank, regV1, regV2, regVG, ccdIters)

      val newModel = model.leftOuterJoin(solutions)
        .map { case (index, ((linear, factor), solution)) =>
          if (solution.nonEmpty) {
            (index, (linear, solution.get))
          } else {
            (index, (linear, factor))
          }
        }

      (newModel, true)
    }
  }


  /**
    * input
    * 00           instance_index   Long
    * 01           instance_label   Float
    * 02           instance_weight  Float
    * 03           indices          Array[Long]     non-zero indices
    * 04           values           Array[Float]
    *
    * model
    * 00           index            Long            feature index
    * 01           linear           Float
    * 02           factor           Array[Float]
    * 03           selected         Bool
    */
  def predictAndFlatten[@spec(Float, Double) V: Fractional : ClassTag](instances: RDD[(Long, V, V, Array[Long], Array[V])],
                                                                       intercept: V,
                                                                       model: RDD[(Long, (V, Array[V]))],
                                                                       rank: Int,
                                                                       selector: IndicesSelector): RDD[(Long, (V, V, V, V, Array[V]))] = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    val negativeInfinity = -one / zero

    instances.flatMap { case (instanceIndex, instanceLabel, instanceWeight, indices, values) =>
      val selectedIndices = indices.filter(selector.contain)

      if (selectedIndices.nonEmpty) {
        var first = true
        indices.iterator
          .zip(values.iterator)
          .filter(_._2 != zero)
          .map { case (index, value) =>
            if (first) {
              first = false
              (index, (instanceIndex, instanceLabel, instanceWeight, value, selectedIndices, indices, values))
            } else {
              (index, (instanceIndex, instanceLabel, instanceWeight, value, Array.emptyLongArray, Array.emptyLongArray, Array.empty[V]))
            }
          }
      } else {
        Iterator.empty
      }

    }.join(model)

      .map { case (_, ((instanceIndex, instanceLabel, instanceWeight, value, selectedIndices, indices, values), (linear, factor))) =>
        (instanceIndex, (instanceLabel, instanceWeight, value, selectedIndices, indices, values, linear, factor))

      }.aggregateByKey((new PredictWithDotsAggregator[V](rank), negativeInfinity, negativeInfinity, Array.emptyLongArray, Array.emptyLongArray, Array.empty[V]))(
      seqOp = {
        case ((predAgg, instanceLabel_, instanceWeight_, selectedIndices_, indices_, values_),
        (instanceLabel, instanceWeight, value, selectedIndices, indices, values, linear, factor)) =>
          (predAgg.update(value, linear, factor),
            max(instanceLabel, instanceLabel_),
            max(instanceWeight, instanceWeight_),
            Seq(selectedIndices, selectedIndices_).maxBy(_.length),
            Seq(indices, indices_).maxBy(_.length),
            Seq(values, values_).maxBy(_.length))

      }, combOp = {
        case ((predAgg1, instanceLabel1, instanceWeight1, selectedIndices1, indices1, values1),
        (predAgg2, instanceLabel2, instanceWeight2, selectedIndices2, indices2, values2)) =>
          (predAgg1.merge(predAgg2),
            max(instanceLabel1, instanceLabel2),
            max(instanceWeight1, instanceWeight2),
            Seq(selectedIndices1, selectedIndices2).maxBy(_.length),
            Seq(indices1, indices2).maxBy(_.length),
            Seq(values1, values2).maxBy(_.length))

      }).mapPartitions { it =>
      val indicesBuilder = mutable.ArrayBuilder.make[Long]
      val valuesBuilder = mutable.ArrayBuilder.make[V]
      var s = 0
      var i = 0

      it.flatMap { case (_, (predAgg, instanceLabel, instanceWeight, selectedIndices, indices, values)) =>
        indicesBuilder.clear()
        valuesBuilder.clear()
        s = 0
        i = 0

        val (prediction, dots) = predAgg.compute(intercept)

        while (s < selectedIndices.length && i < indices.length) {
          if (selectedIndices(s) == indices(i)) {
            indicesBuilder += indices(i)
            valuesBuilder += values(i)
            s += 1
            i += 1
          } else if (selectedIndices(s) < indices(i)) {
            s += 1
          } else {
            i += 1
          }
        }

        indicesBuilder.result().iterator
          .zip(valuesBuilder.result().iterator)
          .map { case (index, value) =>
            (index, (instanceLabel, instanceWeight, value, prediction, dots))
          }
      }
    }
  }


  /**
    * problems
    * 00           index            Long
    * 01           instance_weight  Float
    * 02           prev_solution    Array[Float]
    * 03           problem_yx       Array[Float]       [y, x0, x1, ...]
    */
  def solve[@spec(Float, Double) V: Fractional : ClassTag](problems: RDD[(Long, (V, V, Array[V], Array[V]))],
                                                           k: Int,
                                                           regL1: V,
                                                           regL2: V,
                                                           regLG: V,
                                                           iters: Int): RDD[(Long, Array[V])] = {
    val fracV = implicitly[Fractional[V]]
    import fracV._

    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    //    problems.map { case (index, (instanceWeight, y, x, factor)) =>
    //      (index, instanceWeight.toFloat, y.toFloat, x.map(_.toFloat), factor.map(_.toFloat))
    //    }.toDF("index", "instanceWeight", "y", "x", "factor").sort("index").show(10, false)


    problems.aggregateByKey((new SolverAggregator[V](k), Array.empty[V]))(
      seqOp = {
        case ((solverAgg, prevSolution_), (instanceWeight, y, x, prevSolution)) =>
          (solverAgg.update(instanceWeight, x, y),
            Seq(prevSolution, prevSolution_).maxBy(_.length))
      },
      combOp = {
        case ((solverAgg1, prevSolution1), (solverAgg2, prevSolution2)) =>
          (solverAgg1.merge(solverAgg2),
            Seq(prevSolution1, prevSolution2).maxBy(_.length))

      }).map { case (index, (solverAgg, prevSolution)) =>
      (index, solverAgg.solve(regL1, regL2, regLG, prevSolution, iters))
    }
  }
}


class PredictWithDotsAggregator[@spec(Float, Double) V: Fractional : ClassTag](val rank: Int) extends Serializable {
  require(rank > 0)

  val fracV = implicitly[Fractional[V]]

  import fracV._

  var linearSum = zero
  val dotSum = Array.fill(rank)(zero)
  val dot2Sum = Array.fill(rank)(zero)

  def update(value: V, linear: V, factor: Array[V]): PredictWithDotsAggregator[V] = {
    if (!fracV.equiv(value, zero)) {
      if (!fracV.equiv(linear, zero)) {
        linearSum = linearSum + value * linear
      }

      if (factor.nonEmpty) {
        require(factor.length == rank)

        var i = 0
        while (i < rank) {
          val s = value * factor(i)
          dotSum(i) = dotSum(i) + s
          dot2Sum(i) = dot2Sum(i) + s * s
          i += 1
        }
      }
    }

    this
  }

  def merge(o: PredictWithDotsAggregator[V]): PredictWithDotsAggregator[V] = {
    linearSum = linearSum + o.linearSum

    var i = 0
    while (i < rank) {
      dotSum(i) = dotSum(i) + o.dotSum(i)
      dot2Sum(i) = dot2Sum(i) + o.dot2Sum(i)
      i += 1
    }

    this
  }

  def compute(intercept: V): (V, Array[V]) = {
    var pred = intercept
    var i = 0
    while (i < rank) {
      pred = pred + (dotSum(i) * dotSum(i) - dot2Sum(i)) / fracV.fromInt(2)
      i += 1
    }

    (pred, dotSum)
  }
}


class DistributedSFMModel private[ml](override val uid: String,
                                      val intercept: Float,
                                      @transient val model: DataFrame)
  extends Model[DistributedSFMModel] with DistributedFMParams with MLWritable {

  override def transform(dataset: Dataset[_]): DataFrame = {
    import DistributedFM._

    val spark = dataset.sparkSession
    import spark.implicits._

    val predUDAF = new predictWithDotsUDAF($(rank), intercept)

    dataset.select(col($(instanceIndexCol)).cast(LongType).as(INSTANCE_INDEX),
      col($(featureIndicesCol)).cast(ArrayType(LongType)).as(INDICES),
      col($(featureValuesCol)).cast(ArrayType(FloatType)).as(VALUES))
      .as[(Long, Array[Long], Array[Float])]

      .flatMap { case (instanceIndex, indices, values) =>
        indices.iterator
          .zip(values.iterator)
          .filter(t => t._2 != 0)
          .map { case (index, value) =>
            (instanceIndex, index, value)
          }
      }.toDF(INSTANCE_INDEX, INDEX, VALUE)

      .join(model.hint("broadcast"), Seq(INDEX))

      .groupBy(INSTANCE_INDEX)

      .agg(predUDAF(col(VALUE), col(LINEAR), col(FACTOR)).as(PREDICTION_DOTS))
      .select(INSTANCE_INDEX, PREDICTION_DOTS)
      .as[(Long, Array[Float])]
      .map(t => (t._1, t._2.head))
      .toDF(INSTANCE_INDEX, $(predictionCol))

      .join(dataset, Seq(INSTANCE_INDEX))
  }

  override def copy(extra: ParamMap): DistributedSFMModel = {
    val copied = new DistributedSFMModel(uid, intercept, model)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new DistributedSFMModel.DistributedSFMModelWriter(this)


  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}


object DistributedSFMModel extends MLReadable[DistributedSFMModel] {

  import DistributedFM._

  override def read: MLReader[DistributedSFMModel] = new DistributedSFMModelReader


  override def load(path: String): DistributedSFMModel = super.load(path)

  private[DistributedSFMModel] class DistributedSFMModelWriter(instance: DistributedSFMModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      val spark = instance.model.sparkSession
      import spark.implicits._

      DefaultParamsWriter.saveMetadata(instance, path, sc)

      val modelPath = new Path(path, "model").toString
      val interceptDF = Seq((-1L, instance.intercept, Array.emptyFloatArray)).toDF(INDEX, LINEAR, FACTOR)

      instance.model
        .union(interceptDF)
        .write.parquet(modelPath)
    }
  }

  private class DistributedSFMModelReader extends MLReader[DistributedSFMModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[DistributedSFMModel].getName

    override def load(path: String): DistributedSFMModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val modelPath = new Path(path, "model").toString
      val df = sparkSession.read.parquet(modelPath)

      val intercept = df.select(LINEAR)
        .where(col(INDEX).equalTo(-1L))
        .head().getFloat(0)

      val modelDF = df.where(col(INDEX).geq(0L))

      val model = new DistributedSFMModel(metadata.uid, intercept, modelDF)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

}
