package org.apache.spark.ml.fm

import scala.collection.mutable

import org.apache.spark.ml.param._
import org.apache.spark.rdd._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel


class DistributedFM extends FM {

  override val uid: String = "distributed_fm"

  override def fit(df: DataFrame): FMModel = ???

  override def copy(extra: ParamMap): LocalFM = defaultCopy(extra)
}


object DistributedFM extends Serializable {

  import Utils._


  def train(input: DataFrame,
            rank: Int,
            maxIters: Int,
            numFeatures: Long,
            numGroups: Int,
            regI1: Double,
            regI2: Double,
            regW1: Double,
            regW2: Double,
            regV1: Double,
            regV2: Double,
            regVG: Double,
            ccdIters: Int,
            std: Double,
            seed: Long,
            checkpointPath: String,
            initialIntercept: Option[Double],
            initialModel: Option[DataFrame]): (Double, DataFrame) = {
    val spark = input.sparkSession

    var intercept: Double = initialIntercept.getOrElse(0.0)
    var model = initialModel.getOrElse {
      spark.range(numFeatures).toDF(INDEX)
        .withColumn(WEIGHT, lit(0.0))
        .withColumn(FACTOR, array(Array.range(0, rank).map(i => randn(seed + i)): _*))
    }

    model.show()

    var iCheck = 0

    var iter = 0
    while (iter < maxIters) {
      println(s"Iteration $iter")

      {
        println(s"update itercept")
        intercept = updateIntercept(input, intercept, model, rank, regI1, regI2)
      }


      {
        model = model.select(INDEX, WEIGHT, FACTOR)
          .sort(INDEX)
          .withColumn(RANDOM, (rand(seed + iter) * numGroups).cast(IntegerType))

        //        val path = s"${checkpointPath}/${iCheck}"
        //        iCheck += 1
        //        model.select(INDEX, WEIGHT, FACTOR)
        //          .sort(INDEX)
        //          .withColumn(RANDOM, (rand(seed + iter) * numGroups).cast(IntegerType))
        //          .write
        //          .mode(SaveMode.Overwrite)
        //          .parquet(path)
        //        model = spark.read.parquet(path)

        var group = 0
        while (group < numGroups) {
          println(s"update weights $group")

          val selModel = model.withColumn(SELECTED,
            when(col(RANDOM).equalTo(group), true).otherwise(false))

          model = updateWeights(input, intercept, selModel, rank, regW1, regW2)

          val path = s"${checkpointPath}/${iCheck}"
          iCheck += 1
          model.write
            .mode(SaveMode.Overwrite)
            .parquet(path)
          model = spark.read.parquet(path)

          group += 1
        }
      }


      {
        model = model.select(INDEX, WEIGHT, FACTOR)
          .sort(INDEX)
          .withColumn(RANDOM, (rand(-seed - iter) * numGroups).cast(IntegerType))


        //        val path = s"${checkpointPath}/${iCheck}"
        //        iCheck += 1
        //        model.select(INDEX, WEIGHT, FACTOR)
        //          .sort(INDEX)
        //          .withColumn(RANDOM, (rand(-seed - iter) * numGroups).cast(IntegerType))
        //          .write
        //          .mode(SaveMode.Overwrite)
        //          .parquet(path)
        //        model = spark.read.parquet(path)

        var group = 0
        while (group < numGroups) {
          println(s"update factors $group")

          val selModel = model.withColumn(SELECTED,
            when(col(RANDOM).equalTo(group), true).otherwise(false))

          model = updateFactors(input, intercept, selModel, rank, regV1, regV2, regVG, ccdIters)

          val path = s"${checkpointPath}/${iCheck}"
          iCheck += 1
          model.write
            .mode(SaveMode.Overwrite)
            .parquet(path)
          model = spark.read.parquet(path)

          group += 1
        }
      }


      iter += 1
    }

    (intercept, model.select(INDEX, WEIGHT, FACTOR))
  }


  /**
    * input
    * 00           instance_index   long
    * 01           instance_label   double
    * 02           instance_weight  double
    * 03           indices          array[long]     non-zero indices
    * 04           values           array[double]
    *
    * model
    * 00           index            long            feature index
    * 01           weight           double
    * 02           factor           array[double]
    */
  def updateIntercept(input: DataFrame,
                      intercept: Double,
                      model: DataFrame,
                      rank: Int,
                      regI1: Double,
                      regI2: Double): Double = {
    val spark = input.sparkSession
    import spark.implicits._

    val predUDAF = new predictWithDotsUDAF(rank, intercept)

    val (err, cnt) = input.select(INSTANCE_INDEX, INSTANCE_LABEL, INSTANCE_WEIGHT, INDICES, VALUES)
      .as[(Long, Double, Double, Array[Long], Array[Double])]
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
        predUDAF(col(VALUE), col(WEIGHT), col(FACTOR)).as(PREDICTION_DOTS))
      .select(INSTANCE_LABEL, INSTANCE_WEIGHT, PREDICTION_DOTS)
      .as[(Double, Double, Array[Double])]
      .map { case (instanceLabel, instanceWeight, predDots) =>
        ((instanceLabel - predDots.head) * instanceWeight, instanceWeight)
      }.toDF(ERROR, INSTANCE_WEIGHT)
      .select(sum(ERROR), sum(INSTANCE_WEIGHT))
      .as[(Double, Double)]
      .head()

    val a = 2 * (cnt + regI2)
    val c = 2 * (err + intercept * cnt)

    if (c + regI1 < 0) {
      (c + regI1) / a
    } else if (c - regI1 > 0) {
      (c - regI1) / a
    } else {
      0.0
    }
  }


  /**
    * input
    * 00           instance_index   long
    * 01           instance_label   double
    * 02           instance_weight  double
    * 03           indices          array[long]     non-zero indices
    * 04           values           array[double]
    *
    * model
    * 00           index            long            feature index
    * 01           weight           double
    * 02           factor           array[double]
    * 03           selected         bool
    */
  def updateWeights(input: DataFrame,
                    intercept: Double,
                    model: DataFrame,
                    rank: Int,
                    regW1: Double,
                    regW2: Double): DataFrame = {
    val spark = input.sparkSession
    import spark.implicits._

    val predicted = predictAndFlatten(input, intercept, model, rank)

    val problems = predicted
      .join(model.select(INDEX, WEIGHT).where(col(SELECTED)).hint("broadcast"), INDEX)
      .select(INSTANCE_LABEL, INSTANCE_WEIGHT, INDEX, VALUE, PREDICTION_DOTS, WEIGHT)
      .as[(Double, Double, Long, Double, Array[Double], Double)]
      .map { case (instanceLabel, instanceWeight, index, value, predDots, weight) =>
        val y = instanceLabel - predDots.head + value * weight
        (index, instanceWeight, Array(y, value), Array(0.0))
      }.toDF(INDEX, INSTANCE_WEIGHT, PROBLEM_YX, PREV_SOLUTION)

    val solutions = solve(problems, 1, regW1, regW2, 0.0, 1)
      .select(INDEX, SOLUTION)
      .as[(Long, Array[Double])]
      .map { case (index, solution) =>
        (index, solution.head)
      }.toDF(INDEX, SOLUTION)

    model.join(solutions.hint("broadcast"), Seq(INDEX), "outer")
      .withColumn(WEIGHT, when(col(SOLUTION).isNotNull, col(SOLUTION)).otherwise(col(WEIGHT)))
      .drop(SOLUTION)
  }


  /**
    * input
    * 00           instance_index   long
    * 01           instance_label   double
    * 02           instance_weight  double
    * 03           indices          array[long]     non-zero indices
    * 04           values           array[double]
    *
    * model
    * 00           index            long            feature index
    * 01           weight           double
    * 02           factor           array[double]
    * 03           selected         bool
    */
  def updateFactors(input: DataFrame,
                    intercept: Double,
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
      .as[(Double, Double, Long, Double, Array[Double], Array[Double])]
      .map { case (instanceLabel, instanceWeight, index, value, predDots, factor) =>
        val yx = Array.ofDim[Double](1 + rank)
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
    * 00           instance_index   long
    * 01           instance_label   double
    * 02           instance_weight  double
    * 03           indices          array[long]     non-zero indices
    * 04           values           array[double]
    *
    * model
    * 00           index            long            feature index
    * 01           weight           double
    * 02           factor           array[double]
    * 03           selected         bool
    */
  def predictAndFlatten(input: DataFrame,
                        intercept: Double,
                        model: DataFrame,
                        rank: Int): DataFrame = {
    val spark = input.sparkSession
    import spark.implicits._

    val arrayNonEmptyUDF = udf { array: Seq[_] => array.nonEmpty }
    val predUDAF = new predictWithDotsUDAF(rank, intercept)

    input.select(INSTANCE_INDEX, INSTANCE_LABEL, INSTANCE_WEIGHT, INDICES, VALUES)
      .as[(Long, Double, Double, Array[Long], Array[Double])]
      .flatMap { case (instanceIndex, instanceLabel, instanceWeight, indices, values) =>
        indices.iterator
          .zip(values.iterator)
          .filter(t => t._2 != 0)
          .map { case (index, value) =>
            (instanceIndex, instanceLabel, instanceWeight, index, value, indices, values)
          }
      }.toDF(INSTANCE_INDEX, INSTANCE_LABEL, INSTANCE_WEIGHT, INDEX, VALUE, INDICES, VALUES)

      .join(model.hint("broadcast"), Seq(INDEX))

      .groupBy(INSTANCE_INDEX)
      .agg(first(INSTANCE_LABEL).as(INSTANCE_LABEL),
        first(INSTANCE_WEIGHT).as(INSTANCE_WEIGHT),
        first(col(INDICES)).as(INDICES),
        first(col(VALUES)).as(VALUES),
        collect_list(when(col(SELECTED), col(INDEX))).as(SELECTED_INDICES),
        predUDAF(col(VALUE), col(WEIGHT), col(FACTOR)).as(PREDICTION_DOTS))
      .select(INSTANCE_LABEL, INSTANCE_WEIGHT, INDICES, VALUES, SELECTED_INDICES, PREDICTION_DOTS)
      .as[(Double, Double, Array[Long], Array[Double], Array[Long], Array[Double])]

      .mapPartitions { it =>
        val buffer = mutable.ListBuffer.empty[(Long, Double)]
        var s = 0
        var i = 0

        it.flatMap { case (instanceLabel, instanceWeight, indices, values, selectedIndices, predDots) =>
          buffer.clear()
          s = 0
          i = 0

          val sortedIndices = selectedIndices.sorted
          while (s < sortedIndices.length && i < indices.length) {
            if (sortedIndices(s) == indices(i)) {
              buffer.append((indices(i), values(i)))
              s += 1
              i += 1
            } else if (sortedIndices(s) < indices(i)) {
              s += 1
            } else {
              i += 1
            }
          }

          buffer.result()
            .iterator
            .map { case (index, value) =>
              (instanceLabel, instanceWeight, index, value, predDots)
            }
        }
      }.toDF(INSTANCE_LABEL, INSTANCE_WEIGHT, INDEX, VALUE, PREDICTION_DOTS)


    //    input.select(INSTANCE_INDEX, INSTANCE_LABEL, INSTANCE_WEIGHT, INDICES, VALUES)
    //      .as[(Long, Double, Double, Array[Long], Array[Double])]
    //      .flatMap { case (instanceIndex, instanceLabel, instanceWeight, indices, values) =>
    //        indices.iterator
    //          .zip(values.iterator)
    //          .filter(t => t._2 != 0)
    //          .map { case (index, value) =>
    //            (instanceIndex, instanceLabel, instanceWeight, index, value, indices, values)
    //          }
    //      }.toDF(INSTANCE_INDEX, INSTANCE_LABEL, INSTANCE_WEIGHT, INDEX, VALUE, INDICES, VALUES)
    //
    //      .join(model.hint("broadcast"), Seq(INDEX))
    //
    //      .groupBy(INSTANCE_INDEX)
    //      .agg(first(INSTANCE_LABEL).as(INSTANCE_LABEL),
    //        first(INSTANCE_WEIGHT).as(INSTANCE_WEIGHT),
    //        first(INDICES).as(INDICES),
    //        first(VALUES).as(VALUES),
    //        collect_list(when(col(SELECTED), col(INDEX))).as(SELECTED_INDICES),
    //        predUDAF(col(VALUE), col(WEIGHT), col(FACTOR)).as(PREDICTION_DOTS))
    //      .select(INSTANCE_LABEL, INSTANCE_WEIGHT, INDICES, VALUES, SELECTED_INDICES, PREDICTION_DOTS)
    //      .as[(Double, Double, Array[Long], Array[Double], Array[Long], Array[Double])]
    //
    //      .mapPartitions { it =>
    //        val buffer = mutable.ListBuffer.empty[(Long, Double)]
    //        var s = 0
    //        var i = 0
    //
    //        it.flatMap { case (instanceLabel, instanceWeight, indices, values, selectedIndices, predDots) =>
    //          buffer.clear()
    //          s = 0
    //          i = 0
    //
    //          val sortedIndices = selectedIndices.sorted
    //          while (s < sortedIndices.length && i < indices.length) {
    //            if (sortedIndices(s) == indices(i)) {
    //              buffer.append((indices(i), values(i)))
    //              s += 1
    //              i += 1
    //            } else if (sortedIndices(s) < indices(i)) {
    //              s += 1
    //            } else {
    //              i += 1
    //            }
    //          }
    //
    //          buffer.result()
    //            .iterator
    //            .map { case (index, value) =>
    //              (instanceLabel, instanceWeight, index, value, predDots)
    //            }
    //        }
    //      }.toDF(INSTANCE_LABEL, INSTANCE_WEIGHT, INDEX, VALUE, PREDICTION_DOTS)

    //    input.select(INSTANCE_INDEX, INDICES, VALUES)
    //      .as[(Long, Array[Long], Array[Double])]
    //      .flatMap { case (instanceIndex, indices, values) =>
    //        indices.iterator
    //          .zip(values.iterator)
    //          .filter(t => t._2 != 0)
    //          .map { case (index, value) =>
    //            (instanceIndex, index, value)
    //          }
    //      }.toDF(INSTANCE_INDEX, INDEX, VALUE)
    //      .join(broadcast(model), Seq(INDEX))
    //      .groupBy(INSTANCE_INDEX)
    //      .agg(collect_list(when(col(SELECTED), col(INDEX))).as(SELECTED_INDICES),
    //        predUDAF(col(VALUE), col(WEIGHT), col(FACTOR)).as(PREDICTION_DOTS))
    //      .join(input, INSTANCE_INDEX)
    //      .select(INSTANCE_LABEL, INSTANCE_WEIGHT, INDICES, VALUES, SELECTED_INDICES, PREDICTION_DOTS)
    //      .as[(Double, Double, Array[Long], Array[Double], Array[Long], Array[Double])]
    //      .flatMap { case (instanceLabel, instanceWeight, indices, values, selectedIndices, predDots) =>
    //        val set = selectedIndices.toSet
    //        indices.iterator
    //          .zip(values.iterator)
    //          .filter(t => set.contains(t._1))
    //          .map { case (index, value) =>
    //            (instanceLabel, instanceWeight, index, value, predDots)
    //          }
    //      }.toDF(INSTANCE_LABEL, INSTANCE_WEIGHT, INDEX, VALUE, PREDICTION_DOTS)
  }


  /**
    * problems
    * 00           index            long
    * 01           instance_weight  double
    * 02           prev_solution    array[double]
    * 03           problem_yx       array[double]       [y, x0, x1, ...]
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
      .as[(Long, Array[Double], Array[Double])]
      .map { case (index, stat, prevSolution) =>
        val solution = Utils.solve[Double](stat, k, regL1, regL2, regLG, prevSolution, iters)
        (index, solution)
      }.toDF(INDEX, SOLUTION)
  }
}


class predictWithDotsUDAF(val rank: Int,
                          val intercept: Double) extends UserDefinedAggregateFunction {

  override def inputSchema: StructType = StructType(
    StructField("value", DoubleType, false) ::
      StructField("weight", DoubleType, false) ::
      StructField("factor", ArrayType(DoubleType, false), false) :: Nil
  )

  override def bufferSchema: StructType = StructType(
    StructField("w_sum", DoubleType, false) ::
      StructField("dot_sum", ArrayType(DoubleType, false), false) ::
      StructField("dot2_sum", ArrayType(DoubleType, false), false) :: Nil
  )

  override def dataType: DataType = ArrayType(DoubleType, false)

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = 0.0
    buffer(1) = Array.ofDim[Double](rank)
    buffer(2) = Array.ofDim[Double](rank)
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val v = input.getDouble(0)

    if (v != 0) {
      val weight = input.getDouble(1)
      if (weight != 0) {
        buffer(0) = buffer.getDouble(0) + v * weight
      }

      val factor = input.getSeq[Double](2).toArray
      if (factor.nonEmpty) {
        require(factor.length == rank)

        val dots = buffer.getSeq[Double](1).toArray
        val dots2 = buffer.getSeq[Double](2).toArray
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

    buffer1(0) = buffer1.getDouble(0) + buffer2.getDouble(0)

    val dots_a = buffer1.getSeq[Double](1).toArray
    val dots2_a = buffer1.getSeq[Double](2).toArray

    val dots_b = buffer2.getSeq[Double](1).toArray
    val dots2_b = buffer2.getSeq[Double](2).toArray

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

    var pred = intercept + buffer.getDouble(0)

    val dots = buffer.getSeq[Double](1).toArray
    val dots2 = buffer.getSeq[Double](2).toArray

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
    StructField("weight", DoubleType, false) ::
      StructField("yx", ArrayType(DoubleType, false), false) :: Nil
  )

  override def bufferSchema: StructType = StructType(
    StructField("stat", ArrayType(DoubleType, false), false) :: Nil
  )

  override def dataType: DataType = ArrayType(DoubleType, false)

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = Utils.initStat[Double](k)
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val w = input.getDouble(0)

    val p = input.getSeq[Double](1).toArray
    val y = p.head
    val x = p.tail
    require(x.length == k)

    val stat = buffer.getSeq[Double](0).toArray
    Utils.updateStat[Double](stat, k, w, x, y)
    buffer(0) = stat
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    val stat1 = buffer1.getSeq[Double](0).toArray
    val stat2 = buffer2.getSeq[Double](0).toArray
    Utils.mergeStat[Double](stat1, stat2)
    buffer1(0) = stat1
  }

  override def evaluate(buffer: Row): Any = {
    buffer.getSeq[Double](0)
  }
}

