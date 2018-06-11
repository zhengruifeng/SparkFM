package org.apache.spark.ml.fm

import org.apache.spark.ml.param._
import org.apache.spark.sql._

abstract class FM extends FMParams {
  def fit(dataset: Dataset[_]): FMModel
}


class LocalFM extends FM {

  override val uid: String = "local_fm"

  override def fit(dataset: Dataset[_]): FMModel = ???

  override def copy(extra: ParamMap): LocalFM = defaultCopy(extra)
}


