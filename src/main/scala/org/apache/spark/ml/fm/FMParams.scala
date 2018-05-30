package org.apache.spark.ml.fm

import scala.util.Try

import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.storage.StorageLevel

private[fm] trait FMParams extends Params with HasLabelCol with HasWeightCol with HasFeaturesCol
  with HasFitIntercept with HasRegParam with HasElasticNetParam with HasMaxIter with HasSeed
  with HasCheckpointInterval {

  /**
    * The length of factor vectors.
    * (default = 4)
    */
  final val k = new IntParam(this, "rank",
    "The length of factor vectors. (> 0)", ParamValidators.gt(0))

  def getRank: Int = $(k)

  setDefault(k -> 4)


  /**
    * Std for initialization of factors.
    * (default = 0.1)
    */
  final val initStd = new DoubleParam(this, "initStd",
    "Std for initialization of factors. (> 0)", ParamValidators.gt(0))

  def getInitStd: Double = $(initStd)

  setDefault(initStd -> 0.1)


  /**
    * Group regularization parameter of factors.
    * (default = 0.0)
    */
  final val regFactor = new DoubleParam(this, "regFactor",
    "Group regularization parameter of factors. (>= 0)", ParamValidators.gtEq(0))

  def getRegFactor: Double = $(regFactor)

  setDefault(regFactor -> 0.0)


  /**
    * Whether to fit weights.
    * (default = true)
    */
  final val fitWeights: BooleanParam = new BooleanParam(this, "fitWeights",
    "Whether to fit weights")

  def getFitWeights: Boolean = $(fitWeights)

  setDefault(fitWeights, true)


  /**
    * Maximum number of iterations for each sub-problem.
    * (default = 5)
    */
  final val maxSubIters: IntParam = new IntParam(this, "maxSubIters",
    "Maximum number of iterations for each sub-problem. (> 0)",
    ParamValidators.gt(0))

  def getMaxSubIters: Int = $(maxSubIters)

  setDefault(maxSubIters, 5)


  /**
    * Orthogonal splits of feature space.
    */
  final val splits: IntArrayParam = new IntArrayParam(this, "splits",
    "Orthogonal splits of feature space.",
    (arr: Array[Int]) => {
      arr.nonEmpty &&
        arr.forall(_ >= 0) &&
        arr.distinct.length == arr.length &&
        (0 until arr.length - 1).forall { i => arr(i) < arr(i + 1) }
    })

  def getSplits: Array[Int] = $(splits)


  /**
    * StorageLevel for intermediate datasets. Cannot be "NONE".
    * (default = "MEMORY_AND_DISK")
    */
  val intermediateStorageLevel = new Param[String](this, "intermediateStorageLevel",
    "StorageLevel for intermediate datasets. Cannot be 'NONE'.",
    (s: String) => Try(StorageLevel.fromString(s)).isSuccess && s != "NONE")

  def getIntermediateStorageLevel: String = $(intermediateStorageLevel)

  setDefault(intermediateStorageLevel -> "MEMORY_AND_DISK")


  /**
    * Minimum size of orthogonal group for pre-sorting.
    * (default = 4096)
    */
  val minSortedGroup = new IntParam(this, "minSortedGroup",
    "Minimum size of orthogonal group for pre-sorting. (> 0)",
    ParamValidators.gt(0))

  def getMinSortedGroup: Int = $(minSortedGroup)

  setDefault(minSortedGroup -> 4096)


  /**
    * Whether to treat the model as a distributed one..
    * (default = false)
    */
  final val distributed = new BooleanParam(this, "distributed",
    "Whether to treat the model as a distributed one.")

  def getDistributed: Boolean = $(distributed)

  setDefault(distributed -> false)
}
