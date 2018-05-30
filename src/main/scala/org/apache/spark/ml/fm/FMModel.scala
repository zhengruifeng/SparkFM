package org.apache.spark.ml.fm

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.ml._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

abstract class FMModel(val uid: String)
  extends Model[FMModel] with FMParams with Logging with MLWritable {

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def toLocal: FMModel

  def toDistributed: FMModel

  def isDistributed: Boolean

}


class LocalFMModel(uid: String,
                   val intercept: Double,
                   val weights: Vector,
                   val factors: Matrix) extends FMModel(uid) {

  override def copy(extra: ParamMap): FMModel = ???

  override def transform(dataset: Dataset[_]): DataFrame = ???

  override def transformSchema(schema: StructType): StructType = ???

  override def write: MLWriter = ???

  override def toLocal: FMModel = this

  override def toDistributed: FMModel = ???

  override def isDistributed: Boolean = false
}


class DistributedFMModel(uid: String,
                         val intercept: Double,
                         val weightsAndFactors: DataFrame) extends FMModel(uid) {

  override def copy(extra: ParamMap): FMModel = ???

  override def transform(dataset: Dataset[_]): DataFrame = ???

  override def transformSchema(schema: StructType): StructType = ???

  override def write: MLWriter = ???

  override def toLocal: FMModel = ???

  override def toDistributed: FMModel = this

  override def isDistributed: Boolean = true
}






