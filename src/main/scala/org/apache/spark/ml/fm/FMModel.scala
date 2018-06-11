package org.apache.spark.ml.fm

import org.apache.spark.ml.linalg._
import org.apache.spark.ml._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

trait FMModel {

  def transform(dataset: Dataset[_]): DataFrame

//  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def toLocal: FMModel

  def toDistributed: FMModel

  def isDistributed: Boolean
}


class LocalFMModel(val uid: String,
                   val intercept: Double,
                   val weights: Vector,
                   val factors: Matrix) extends Model[LocalFMModel] with FMParams with FMModel with MLWritable {

  override def copy(extra: ParamMap): LocalFMModel = ???

  override def transform(dataset: Dataset[_]): DataFrame = ???

  override def transformSchema(schema: StructType): StructType = ???

  override def write: MLWriter = ???

  override def toLocal: FMModel = this

  override def toDistributed: FMModel = ???

  override def isDistributed: Boolean = false

}






