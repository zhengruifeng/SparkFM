package org.apache.spark.ml.fm

import org.apache.spark.ml.param._
import org.apache.spark.sql.DataFrame

abstract class FM extends FMParams {
  def fit(df: DataFrame): FMModel
}


class LocalFM extends FM {

  override val uid: String = "local_fm"

  override def fit(df: DataFrame): FMModel = ???

  override def copy(extra: ParamMap): LocalFM = defaultCopy(extra)
}


