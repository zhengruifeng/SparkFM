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
  val regInterceptL1 = new DoubleParam(this, "regInterceptL1", "L1-norm regularization of intercept", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegI1: Double = $(regInterceptL1)

  setDefault(regInterceptL1 -> 0.0)


  /**
    * Param for L2-norm regularization of intercept.
    * Default: 0.0
    *
    * @group param
    */
  val regInterceptL2 = new DoubleParam(this, "regInterceptL2", "L2-norm regularization of intercept", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegI2: Double = $(regInterceptL2)

  setDefault(regInterceptL2 -> 0.0)


  /**
    * Param for L1-norm regularization of linear weights.
    * Default: 0.0
    *
    * @group param
    */
  val regLinearL1 = new DoubleParam(this, "regLinearL1", "L1-norm regularization of linear weights", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegW1: Double = $(regLinearL1)

  setDefault(regLinearL1 -> 0.0)


  /**
    * Param for L2-norm regularization of linear weights.
    * Default: 0.0
    *
    * @group param
    */
  val regLinearL2 = new DoubleParam(this, "regLinearL2", "L2-norm regularization of linear weights", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegW2: Double = $(regLinearL2)

  setDefault(regLinearL2 -> 0.0)


  /**
    * Param for L1-norm regularization of factors.
    * Default: 0.0
    *
    * @group param
    */
  val regFactorL1 = new DoubleParam(this, "regFactorL1", "L1-norm regularization of factors", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegV1: Double = $(regFactorL1)

  setDefault(regFactorL1 -> 0.0)


  /**
    * Param for L2-norm regularization of factors.
    * Default: 0.0
    *
    * @group param
    */
  val regFactorL2 = new DoubleParam(this, "regFactorL2", "L2-norm regularization of factors", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegV2: Double = $(regFactorL2)

  setDefault(regFactorL2 -> 0.0)


  /**
    * Param for L2-norm regularization of factor-groups.
    * Default: 0.0
    *
    * @group param
    */
  val regFactorLG = new DoubleParam(this, "regFactorLG", "L2-norm regularization of factor-groups", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegVG: Double = $(regFactorLG)

  setDefault(regFactorLG -> 0.0)


  /**
    * Param for maximum iterations in base CCD solver.
    * Default: 5
    *
    * @group param
    */
  val maxCCDIters = new IntParam(this, "maxCCDIters", "maximum iterations in base CCD solver", ParamValidators.gtEq(3))

  /** @group getParam */
  def getMaxCCDIters: Int = $(maxCCDIters)

  setDefault(maxCCDIters -> 5)


  /**
    * Param for whether to fit linear coefficients.
    * Default: true
    *
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


  /**
    * Float precision to represent internal numerical types.
    * (default = "float")
    *
    * @group param
    */
  final val floatType: Param[String] =
    new Param[String](this, "floatType", "Float precision to represent internal numerical types.",
      ParamValidators.inArray[String](Array("float", "double")))

  def getFloatType: String = $(floatType)

  setDefault(floatType -> "float")
}


private[fm] trait DistributedFMParams extends FMParams
  with HasCheckpointInterval {

  /**
    * Param for number of features.
    *
    * @group param
    */
  val numFeatures = new LongParam(this, "numFeatures", "number of features", ParamValidators.gt(0))

  /** @group getParam */
  def getNumFeatures: Long = $(numFeatures)


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
    * Default: "featureIndices"
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