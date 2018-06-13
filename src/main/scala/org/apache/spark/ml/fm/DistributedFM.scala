package org.apache.spark.ml.fm

import scala.collection.mutable

import org.apache.hadoop.fs.Path
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._

import org.apache.spark.ml._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel


class DistributedFM extends Estimator[DistributedFMModel]
  with DistributedFMParams with DefaultParamsWritable {

  import Utils._

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setInstanceIndexCol(value: String): this.type = set(instanceIndexCol, value)

  def setFeatureIndiceCol(value: String): this.type = set(featureIndicesCol, value)

  def setFeatureValuesCol(value: String): this.type = set(featureValuesCol, value)

  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setSeed(value: Long): this.type = set(seed, value)

  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  def setRank(value: Int): this.type = set(rank, value)

  def setNumRandGroups(value: Int): this.type = set(numRandGroups, value)

  def setRegI1(value: Double): this.type = set(regI1, value)

  def setRegI2(value: Double): this.type = set(regI2, value)

  def setRegW1(value: Double): this.type = set(regW1, value)

  def setRegW2(value: Double): this.type = set(regW2, value)

  def setRegV1(value: Double): this.type = set(regV1, value)

  def setRegV2(value: Double): this.type = set(regV2, value)

  def setRegVG(value: Double): this.type = set(regVG, value)

  def setNumFeaturesG(value: Long): this.type = set(numFeatures, value)

  def setMaxCCDIters(value: Int): this.type = set(maxCCDIters, value)

  def setInitModelPath(value: String): this.type = set(initModelPath, value)

  def setIntermediateStorageLevel(value: String): this.type = set(intermediateStorageLevel, value)

  def setFinalStorageLevel(value: String): this.type = set(finalStorageLevel, value)


  override val uid: String = "distributed_fm"

  override def fit(dataset: Dataset[_]): DistributedFMModel = {

    require((isDefined(featuresCol) && $(featuresCol).nonEmpty) ||
      (isDefined(featureIndicesCol) && $(featureIndicesCol).nonEmpty
        && isDefined(featureValuesCol) && $(featureValuesCol).nonEmpty))

    val spark = dataset.sparkSession
    import spark.implicits._

    val realNumFeatures = computeNumFeatures(dataset)

    val input = formatInput(dataset)

    val handlePersistence = dataset.storageLevel == StorageLevel.NONE
    if (handlePersistence) {
      input.persist(StorageLevel.fromString($(intermediateStorageLevel)))
    }

    var (intercept, model) = initialize(spark, realNumFeatures)

    val instr = Instrumentation.create(this, dataset)

    var iter = 0
    while (iter < $(maxIter)) {
      instr.log(s"Iteration $iter")

      if ($(fitIntercept)) {
        instr.log(s"update itercept")
        intercept = DistributedFM.updateIntercept(input, intercept, model, $(rank), $(regI1), $(regI2))
      }

      if ($(fitLinear)) {
        model = model.select(INDEX, LINEAR, FACTOR)
          .sort(INDEX)
          .withColumn(RANDOM, (rand($(seed) + iter) * $(numRandGroups)).cast(IntegerType))

        var group = 0
        while (group < $(numRandGroups)) {
          instr.log(s"update weights $group")

          val selModel = model.withColumn(SELECTED,
            when(col(RANDOM).equalTo(group), true).otherwise(false))

          model = DistributedFM.updateLinears(input, intercept, selModel, $(rank), $(regW1), $(regW2))

          val path = s"${$(checkpointDir)}/Iter-${iter}_Linears${group}_FM_Snapshot"

          model.write
            .mode(SaveMode.Overwrite)
            .parquet(path)

          model = spark.read.parquet(path)

          group += 1
        }
      }

      {
        model = model.select(INDEX, LINEAR, FACTOR)
          .sort(INDEX)
          .withColumn(RANDOM, (rand(-$(seed) - iter) * $(numRandGroups)).cast(IntegerType))

        var group = 0
        while (group < $(numRandGroups)) {
          instr.log(s"update factors $group")

          val selModel = model.withColumn(SELECTED,
            when(col(RANDOM).equalTo(group), true).otherwise(false))

          model = DistributedFM.updateFactors(input, intercept, selModel, $(rank), $(regV1), $(regV2), $(regVG), $(maxCCDIters))

          val path = s"${$(checkpointDir)}/Iter-${iter}_Factors${group}_FM_Snapshot"

          model.write
            .mode(SaveMode.Overwrite)
            .parquet(path)

          model = spark.read.parquet(path)

          group += 1
        }
      }

      iter += 1
    }

    val finalModel = model.select(INDEX, LINEAR, FACTOR)
      .as[(Long, Float, Array[Float])]
      .flatMap { case (index, linear, factor) =>
        if (factor.forall(_ == 0)) {
          if (linear == 0) {
            Iterator.empty
          } else {
            Iterator.single(index, linear, Array.emptyFloatArray)
          }
        } else {
          Iterator.single(index, linear, factor)
        }
      }.toDF(INDEX, LINEAR, FACTOR)

    finalModel.persist(StorageLevel.fromString($(finalStorageLevel)))
    finalModel.count()

    if (handlePersistence) {
      input.unpersist(false)
    }

    new DistributedFMModel(uid, intercept, finalModel)
  }

  private def computeNumFeatures(dataset: Dataset[_]): Long = {
    val spark = dataset.sparkSession
    import spark.implicits._

    if ($(numFeatures) > 0) {
      $(numFeatures)

    } else if (isDefined(featuresCol) && $(featuresCol).nonEmpty) {
      dataset.select($(featuresCol)).head()
        .getAs[Vector](0).size.toLong

    } else {
      dataset.select($(featureIndicesCol))
        .as[Array[Long]].rdd
        .map { indices =>
          if (indices.nonEmpty) {
            indices.last
          } else {
            0L
          }
        }.max() + 1
    }
  }

  private def formatInput(dataset: Dataset[_]): DataFrame = {
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
            (indicesBuilder.result(), valuesBuilder.result(), instanceLabel, instanceWeight, instanceIndex)
          }
        }.toDF(INDICES, VALUES, INSTANCE_LABEL, INSTANCE_WEIGHT, INSTANCE_INDEX)

    } else {
      dataset.select(col($(featureIndicesCol)).as(INDICES),
        col($(featureValuesCol)).as(VALUES),
        col($(labelCol)).cast(FloatType).as(INSTANCE_LABEL),
        w.cast(FloatType).as(INSTANCE_WEIGHT),
        col($(instanceIndexCol)).cast(LongType).as(INSTANCE_INDEX))
    }
  }

  private def initialize(spark: SparkSession,
                         numFeatures: Long): (Float, DataFrame) = {
    if (isSet(initModelPath) && $(initModelPath).nonEmpty) {
      val m = DistributedFMModel.load($(initModelPath))
      (m.intercept, m.model)

    } else {
      val randCols = Array.range(0, $(rank))
        .map(i => randn($(seed) + i).cast(FloatType))

      (0.0F,
        spark.range(numFeatures).toDF(INDEX)
          .withColumns(Seq(LINEAR, FACTOR), Seq(lit(0.0F, array(randCols: _*)))))
    }
  }


  override def copy(extra: ParamMap): DistributedFM = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}


object DistributedFM extends Serializable {

  import Utils._

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
  def updateIntercept(input: DataFrame,
                      intercept: Float,
                      model: DataFrame,
                      rank: Int,
                      regI1: Double,
                      regI2: Double): Float = {
    val spark = input.sparkSession
    import spark.implicits._

    val predUDAF = new predictWithDotsUDAF(rank, intercept)

    val (err, cnt) = input.select(INSTANCE_INDEX, INSTANCE_LABEL, INSTANCE_WEIGHT, INDICES, VALUES)
      .as[(Long, Float, Float, Array[Long], Array[Float])]
      .flatMap { case (instanceIndex, instanceLabel, instanceWeight, indices, values) =>
        indices.iterator
          .zip(values.iterator)
          .filter(t => t._2 != 0)
          .map { case (index, value) =>
            (instanceIndex, instanceLabel, instanceWeight, index, value)
          }
      }.toDF(INSTANCE_INDEX, INSTANCE_LABEL, INSTANCE_WEIGHT, INDEX, VALUE)
      .join(model.hint("broadcast"), Seq(INDEX))
      .groupBy(INSTANCE_INDEX)
      .agg(first(col(INSTANCE_LABEL)).as(INSTANCE_LABEL),
        first(col(INSTANCE_WEIGHT)).as(INSTANCE_WEIGHT),
        predUDAF(col(VALUE), col(LINEAR), col(FACTOR)).as(PREDICTION_DOTS))
      .select(INSTANCE_LABEL, INSTANCE_WEIGHT, PREDICTION_DOTS)
      .as[(Float, Float, Array[Float])]
      .map { case (instanceLabel, instanceWeight, predDots) =>
        ((instanceLabel - predDots.head) * instanceWeight, instanceWeight)
      }.toDF(ERROR, INSTANCE_WEIGHT)
      .select(sum(ERROR).cast(FloatType), sum(INSTANCE_WEIGHT).cast(FloatType))
      .as[(Float, Float)]
      .head()

    val a = 2 * (cnt + regI2)
    val c = 2 * (err + intercept * cnt)

    if (c + regI1 < 0) {
      ((c + regI1) / a).toFloat
    } else if (c - regI1 > 0) {
      ((c - regI1) / a).toFloat
    } else {
      0.0F
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
  def updateLinears(input: DataFrame,
                    intercept: Float,
                    model: DataFrame,
                    rank: Int,
                    regW1: Double,
                    regW2: Double): DataFrame = {
    val spark = input.sparkSession
    import spark.implicits._

    val predicted = predictAndFlatten(input, intercept, model, rank)

    val problems = predicted
      .join(model.select(INDEX, LINEAR).where(col(SELECTED)).hint("broadcast"), INDEX)
      .select(INSTANCE_LABEL, INSTANCE_WEIGHT, INDEX, VALUE, PREDICTION_DOTS, LINEAR)
      .as[(Float, Float, Long, Float, Array[Float], Float)]
      .map { case (instanceLabel, instanceWeight, index, value, predDots, weight) =>
        val y = instanceLabel - predDots.head + value * weight
        (index, instanceWeight, Array(y, value), Array(0.0))
      }.toDF(INDEX, INSTANCE_WEIGHT, PROBLEM_YX, PREV_SOLUTION)

    val solutions = solve(problems, 1, regW1, regW2, 0.0, 1)
      .select(INDEX, SOLUTION)
      .as[(Long, Array[Float])]
      .map { case (index, solution) =>
        (index, solution.head)
      }.toDF(INDEX, SOLUTION)

    model.join(solutions.hint("broadcast"), Seq(INDEX), "outer")
      .withColumn(LINEAR, when(col(SOLUTION).isNotNull, col(SOLUTION)).otherwise(col(LINEAR)))
      .drop(SOLUTION)
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
  def updateFactors(input: DataFrame,
                    intercept: Float,
                    model: DataFrame,
                    rank: Int,
                    regV1: Double,
                    regV2: Double,
                    regVG: Double,
                    ccdIters: Int): DataFrame = {
    val spark = input.sparkSession
    import spark.implicits._

    val predicted = predictAndFlatten(input, intercept, model, rank)

    val problems = predicted
      .join(model.select(INDEX, FACTOR).where(col(SELECTED)).hint("broadcast"), INDEX)
      .select(INSTANCE_LABEL, INSTANCE_WEIGHT, INDEX, VALUE, PREDICTION_DOTS, FACTOR)
      .as[(Float, Float, Long, Float, Array[Float], Array[Float])]
      .map { case (instanceLabel, instanceWeight, index, value, predDots, factor) =>
        val yx = Array.ofDim[Float](1 + rank)
        yx(0) = instanceLabel - predDots.head
        for (f <- 0 until rank) {
          val vfl = factor(f)
          val r = value * (predDots(f + 1) - vfl * value)
          yx(f + 1) = r
          yx(0) += vfl * r
        }
        (index, instanceWeight, yx, factor)
      }.toDF(INDEX, INSTANCE_WEIGHT, PROBLEM_YX, PREV_SOLUTION)

    val solutions = solve(problems, rank, regV1, regV2, regVG, ccdIters)

    model.join(solutions.hint("broadcast"), Seq(INDEX), "outer")
      .withColumn(FACTOR, when(col(SOLUTION).isNotNull, col(SOLUTION)).otherwise(col(FACTOR)))
      .drop(SOLUTION)
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
  def predictAndFlatten(input: DataFrame,
                        intercept: Float,
                        model: DataFrame,
                        rank: Int): DataFrame = {
    val spark = input.sparkSession
    import spark.implicits._

    val predUDAF = new predictWithDotsUDAF(rank, intercept)

    input.select(INSTANCE_INDEX, INSTANCE_LABEL, INSTANCE_WEIGHT, INDICES, VALUES)
      .as[(Long, Float, Float, Array[Long], Array[Float])]
      .flatMap { case (instanceIndex, instanceLabel, instanceWeight, indices, values) =>
        var first = true
        indices.iterator
          .zip(values.iterator)
          .filter(t => t._2 != 0)
          .map { case (index, value) =>
            if (first) {
              first = false
              (instanceIndex, instanceLabel, instanceWeight, index, value, indices, values)
            } else {
              (instanceIndex, instanceLabel, instanceWeight, index, value, null, null)
            }
          }
      }.toDF(INSTANCE_INDEX, INSTANCE_LABEL, INSTANCE_WEIGHT, INDEX, VALUE, INDICES, VALUES)

      .join(model.hint("broadcast"), Seq(INDEX))

      .groupBy(INSTANCE_INDEX)
      .agg(first(INSTANCE_LABEL).as(INSTANCE_LABEL),
        first(INSTANCE_WEIGHT).as(INSTANCE_WEIGHT),
        first(col(INDICES), true).as(INDICES),
        first(col(VALUES), true).as(VALUES),
        collect_list(when(col(SELECTED), col(INDEX))).as(SELECTED_INDICES),
        predUDAF(col(VALUE), col(LINEAR), col(FACTOR)).as(PREDICTION_DOTS))
      .select(INSTANCE_LABEL, INSTANCE_WEIGHT, INDICES, VALUES, SELECTED_INDICES, PREDICTION_DOTS)
      .as[(Float, Float, Array[Long], Array[Float], Array[Long], Array[Float])]

      .mapPartitions { it =>
        val indicesBuilder = mutable.ArrayBuilder.make[Long]
        val valuesBuilder = mutable.ArrayBuilder.make[Float]
        var s = 0
        var i = 0

        it.flatMap { case (instanceLabel, instanceWeight, indices, values, selectedIndices, predDots) =>
          indicesBuilder.clear()
          valuesBuilder.clear()
          s = 0
          i = 0

          val sortedIndices = selectedIndices.sorted
          while (s < sortedIndices.length && i < indices.length) {
            if (sortedIndices(s) == indices(i)) {
              indicesBuilder += indices(i)
              valuesBuilder += values(i)
              s += 1
              i += 1
            } else if (sortedIndices(s) < indices(i)) {
              s += 1
            } else {
              i += 1
            }
          }

          indicesBuilder.result().iterator
            .zip(valuesBuilder.result().iterator)
            .map { case (index, value) =>
              (instanceLabel, instanceWeight, index, value, predDots)
            }
        }
      }.toDF(INSTANCE_LABEL, INSTANCE_WEIGHT, INDEX, VALUE, PREDICTION_DOTS)
  }


  /**
    * problems
    * 00           index            Long
    * 01           instance_weight  Float
    * 02           prev_solution    Array[Float]
    * 03           problem_yx       Array[Float]       [y, x0, x1, ...]
    */
  def solve(problems: DataFrame,
            k: Int,
            regL1: Double,
            regL2: Double,
            regLG: Double,
            iters: Int): DataFrame = {
    val spark = problems.sparkSession
    import spark.implicits._

    val probStatUDAF = new ProblemStatUDAF(k)

    problems.groupBy(col(INDEX))
      .agg(first(col(PREV_SOLUTION)).as(PREV_SOLUTION),
        probStatUDAF(col(INSTANCE_WEIGHT), col(PROBLEM_YX)).as(STAT))
      .select(INDEX, STAT, PREV_SOLUTION)
      .as[(Long, Array[Float], Array[Float])]
      .map { case (index, stat, prevSolution) =>
        val solution = Utils.solve[Float](stat, k, regL1, regL2, regLG, prevSolution, iters)
        (index, solution)
      }.toDF(INDEX, SOLUTION)
  }
}


class predictWithDotsUDAF(val rank: Int,
                          val intercept: Float) extends UserDefinedAggregateFunction {

  override def inputSchema: StructType = StructType(
    StructField("value", FloatType, false) ::
      StructField("weight", FloatType, false) ::
      StructField("factor", ArrayType(FloatType, false), false) :: Nil
  )

  override def bufferSchema: StructType = StructType(
    StructField("w_sum", FloatType, false) ::
      StructField("dot_sum", ArrayType(FloatType, false), false) ::
      StructField("dot2_sum", ArrayType(FloatType, false), false) :: Nil
  )

  override def dataType: DataType = ArrayType(FloatType, false)

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = 0.0F
    buffer(1) = Array.ofDim[Float](rank)
    buffer(2) = Array.ofDim[Float](rank)
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val v = input.getFloat(0)

    if (v != 0) {
      val weight = input.getFloat(1)
      if (weight != 0) {
        buffer(0) = buffer.getFloat(0) + v * weight
      }

      val factor = input.getSeq[Float](2).toArray
      if (factor.nonEmpty) {
        require(factor.length == rank)

        val dots = buffer.getSeq[Float](1).toArray
        val dots2 = buffer.getSeq[Float](2).toArray
        var i = 0
        while (i < rank) {
          val s = v * factor(i)
          dots(i) += s
          dots2(i) += s * s
          i += 1
        }

        buffer(1) = dots
        buffer(2) = dots2
      }
    }
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {

    buffer1(0) = buffer1.getFloat(0) + buffer2.getFloat(0)

    val dots_a = buffer1.getSeq[Float](1).toArray
    val dots2_a = buffer1.getSeq[Float](2).toArray

    val dots_b = buffer2.getSeq[Float](1).toArray
    val dots2_b = buffer2.getSeq[Float](2).toArray

    var i = 0
    while (i < rank) {
      dots_a(i) += dots_b(i)
      dots2_a(i) += dots2_b(i)
      i += 1
    }

    buffer1(1) = dots_a
    buffer1(2) = dots2_a
  }

  override def evaluate(buffer: Row): Any = {

    var pred = intercept + buffer.getFloat(0)

    val dots = buffer.getSeq[Float](1).toArray
    val dots2 = buffer.getSeq[Float](2).toArray

    var i = 0
    while (i < rank) {
      pred += (dots(i) * dots(i) - dots2(i)) / 2
      i += 1
    }

    Array(pred) ++ dots
  }
}


class ProblemStatUDAF(val k: Int) extends UserDefinedAggregateFunction {
  require(k > 0)

  override def inputSchema: StructType = StructType(
    StructField("weight", FloatType, false) ::
      StructField("yx", ArrayType(FloatType, false), false) :: Nil
  )

  override def bufferSchema: StructType = StructType(
    StructField("stat", ArrayType(FloatType, false), false) :: Nil
  )

  override def dataType: DataType = ArrayType(FloatType, false)

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = Utils.initStat[Float](k)
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val w = input.getFloat(0)

    val p = input.getSeq[Float](1).toArray
    val y = p.head
    val x = p.tail
    require(x.length == k)

    val stat = buffer.getSeq[Float](0).toArray
    Utils.updateStat[Float](stat, k, w, x, y)
    buffer(0) = stat
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    val stat1 = buffer1.getSeq[Float](0).toArray
    val stat2 = buffer2.getSeq[Float](0).toArray
    Utils.mergeStat[Float](stat1, stat2)
    buffer1(0) = stat1
  }

  override def evaluate(buffer: Row): Any = {
    buffer.getSeq[Float](0)
  }
}


class DistributedFMModel private[ml](override val uid: String,
                                     val intercept: Float,
                                     @transient val model: DataFrame)
  extends Model[DistributedFMModel] with DistributedFMParams with MLWritable {

  override def transform(dataset: Dataset[_]): DataFrame = {
    import Utils._

    val spark = dataset.sparkSession
    import spark.implicits._

    val predUDAF = new predictWithDotsUDAF($(rank), intercept)

    dataset.select(col($(featureIndicesCol)).as(INDICES),
      col($(featureValuesCol)).as(VALUES),
      col($(instanceIndexCol)).cast(LongType).as(INSTANCE_INDEX))
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

  override def copy(extra: ParamMap): DistributedFMModel = {
    val copied = new DistributedFMModel(uid, intercept, model)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new DistributedFMModel.DistributedFMModelWriter(this)


  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}


object DistributedFMModel extends MLReadable[DistributedFMModel] {

  import Utils._

  override def read: MLReader[DistributedFMModel] = new DistributedFMModelReader


  override def load(path: String): DistributedFMModel = super.load(path)

  private[DistributedFMModel] class DistributedFMModelWriter(instance: DistributedFMModel) extends MLWriter {

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

  private class DistributedFMModelReader extends MLReader[DistributedFMModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[DistributedFMModel].getName

    override def load(path: String): DistributedFMModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val modelPath = new Path(path, "model").toString
      val df = sparkSession.read.parquet(modelPath)

      val intercept = df.select(LINEAR)
        .where(col(INDEX).equalTo(-1L))
        .head().getFloat(0)

      val modelDF = df.where(col(INDEX).geq(0L))

      val model = new DistributedFMModel(metadata.uid, intercept, modelDF)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

}






