package org.apache.spark.ml.fm

import scala.util.Try

import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel


private[fm] trait FMParams extends Params
  with HasLabelCol with HasWeightCol with HasFeaturesCol with HasPredictionCol
  with HasFitIntercept with HasMaxIter with HasSeed {

  /**
    * Param for rank of the factorization machine (positive).
    * Default: 8
    *
    * @group param
    */
  val rank = new IntParam(this, "rank", "rank of the factorization machine", ParamValidators.gtEq(1))

  /** @group getParam */
  def getRank: Int = $(rank)

  setDefault(rank -> 8)


  /**
    * Param for number of randomized groups (positive).
    * Default: 10
    *
    * @group param
    */
  val numRandGroups = new IntParam(this, "numRandGroups", "number of randomized groups", ParamValidators.gtEq(1))

  /** @group getParam */
  def getNumRandGroups: Int = $(numRandGroups)

  setDefault(numRandGroups -> 10)


  /**
    * Param for L1-norm regularization of intercept.
    * Default: 0.0
    *
    * @group param
    */
  val regI1 = new DoubleParam(this, "regI1", "L1-norm regularization of intercept", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegI1: Double = $(regI1)

  setDefault(regI1 -> 0.0)


  /**
    * Param for L2-norm regularization of intercept.
    * Default: 0.0
    *
    * @group param
    */
  val regI2 = new DoubleParam(this, "regI2", "L2-norm regularization of intercept", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegI2: Double = $(regI2)

  setDefault(regI2 -> 0.0)


  /**
    * Param for L1-norm regularization of linear weights.
    * Default: 0.0
    *
    * @group param
    */
  val regW1 = new DoubleParam(this, "regW1", "L1-norm regularization of linear weights", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegW1: Double = $(regW1)

  setDefault(regW1 -> 0.0)


  /**
    * Param for L2-norm regularization of linear weights.
    * Default: 0.0
    *
    * @group param
    */
  val regW2 = new DoubleParam(this, "regW2", "L2-norm regularization of linear weights", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegW2: Double = $(regW2)

  setDefault(regW2 -> 0.0)


  /**
    * Param for L1-norm regularization of factors.
    * Default: 0.0
    *
    * @group param
    */
  val regV1 = new DoubleParam(this, "regV1", "L1-norm regularization of factors", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegV1: Double = $(regV1)

  setDefault(regV1 -> 0.0)


  /**
    * Param for L2-norm regularization of factors.
    * Default: 0.0
    *
    * @group param
    */
  val regV2 = new DoubleParam(this, "regV2", "L2-norm regularization of factors", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegV2: Double = $(regV2)

  setDefault(regV2 -> 0.0)


  /**
    * Param for L2-norm regularization of factor-groups.
    * Default: 0.0
    *
    * @group param
    */
  val regVG = new DoubleParam(this, "regVG", "L2-norm regularization of factor-groups", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegVG: Double = $(regVG)

  setDefault(regVG -> 0.0)


  /**
    * Param for maximum iterations in base CCD solver.
    * Default: 5
    *
    * @group param
    */
  val maxCCDIters = new IntParam(this, "maxCCDIters", "maximum iterations in base CCD solver", ParamValidators.gtEq(5))

  /** @group getParam */
  def getMaxCCDIters: Int = $(maxCCDIters)

  setDefault(maxCCDIters -> 5)


  /**
    * Param for whether to fit linear coefficients.
    * Default: true
    * @group param
    */
  final val fitLinear: BooleanParam = new BooleanParam(this, "fitLinear", "whether to fit linear coefficients")

  setDefault(fitLinear, true)

  /** @group getParam */
  final def getFitLinear: Boolean = $(fitLinear)


  /**
    * Param for initial model path
    *
    * @group expertParam
    */
  val initModelPath = new Param[String](this, "initModelPath", "initial model path")

  /** @group expertGetParam */
  def getInitModelPath: String = $(initModelPath)
}


private[fm] trait DistributedFMParams extends FMParams
  with HasCheckpointInterval {

  /**
    * Param for number of features.
    * Default: -1
    *
    * @group param
    */
  val numFeatures = new LongParam(this, "numFeatures", "number of features")

  /** @group getParam */
  def getNumFeatures: Long = $(numFeatures)

  setDefault(numFeatures -> -1)


  /**
    * Param for the column name for instance indices.
    * Default: "instanceIndex"
    *
    * @group param
    */
  val instanceIndexCol = new Param[String](this, "instanceIndexCol", "column name for instance indices")

  /** @group getParam */
  def getInstanceIndexCol: String = $(instanceIndexCol)

  setDefault(instanceIndexCol -> "instanceIndex")


  /**
    * Param for the column name for non-zero feature indices.
    * Default: "indices"
    *
    * @group param
    */
  val featureIndicesCol = new Param[String](this, "featureIndicesCol", "column name for non-zero feature indices")

  /** @group getParam */
  def getFeatureIndiceCol: String = $(featureIndicesCol)

  setDefault(featureIndicesCol -> "featureIndices")


  /**
    * Param for the column name for non-zero feature values.
    * Default: "featureValues"
    *
    * @group param
    */
  val featureValuesCol = new Param[String](this, "featureValuesCol", "column name for non-zero feature values")

  /** @group getParam */
  def getFeatureValuesCol: String = $(featureValuesCol)

  setDefault(featureValuesCol -> "featureValues")


  /**
    * Param for directory for model checkpointing.
    *
    * @group param
    */
  val checkpointDir = new Param[String](this, "checkpointDir", "directory for model checkpointing")

  /** @group getParam */
  def getCheckpointDir: String = $(checkpointDir)


  /**
    * Param for StorageLevel for intermediate datasets. Pass in a string representation of
    * `StorageLevel`. Cannot be "NONE".
    * Default: "MEMORY_AND_DISK".
    *
    * @group expertParam
    */
  val intermediateStorageLevel = new Param[String](this, "intermediateStorageLevel",
    "StorageLevel for intermediate datasets. Cannot be 'NONE'.",
    (s: String) => Try(StorageLevel.fromString(s)).isSuccess && s != "NONE")

  /** @group expertGetParam */
  def getIntermediateStorageLevel: String = $(intermediateStorageLevel)

  setDefault(intermediateStorageLevel -> "MEMORY_AND_DISK")


  /**
    * Param for StorageLevel for ALS model factors. Pass in a string representation of
    * `StorageLevel`.
    * Default: "MEMORY_AND_DISK".
    *
    * @group expertParam
    */
  val finalStorageLevel = new Param[String](this, "finalStorageLevel",
    "StorageLevel for ALS model factors.",
    (s: String) => Try(StorageLevel.fromString(s)).isSuccess)

  /** @group expertGetParam */
  def getFinalStorageLevel: String = $(finalStorageLevel)

  setDefault(finalStorageLevel -> "MEMORY_AND_DISK")


  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.appendColumn(schema, $(predictionCol), FloatType)
  }
}