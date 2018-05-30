package org.apache.spark.ml.fm

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.{specialized => spec}

import org.apache.spark.ml.param._
import org.apache.spark.rdd._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel


class DistributedFMRDD {

  import Utils._


}


object DistributedFMRDD extends Serializable {


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
    *
    * ouput
    * 00           index            I            feature index
    * 01           instance_label   H
    * 02           instance_weight  H
    * 03           value            H
    * 04           prediction       H
    * 05           dots             array[H]
    */
  //  def predictAndFlatten[H: Numeric : ClassTag, I: Integral : ClassTag](input: RDD[(Long, H, H, Array[I], Array[H])],
  //                                                                       intercept: Double,
  //                                                                       model: RDD[(I, (H, Array[H], Boolean))],
  //                                                                       rank: Int): RDD[(I, H, H, H, H, Array[H])] = {
  //    val numH = implicitly[Numeric[H]]
  //
  //    input.flatMap { case (instanceIndex, instanceLabel, instanceWeight, indices, values) =>
  //      indices.iterator
  //        .zip(values.iterator)
  //        .filter(t => t._2 != 0)
  //        .map { case (index, value) =>
  //          (index, (instanceIndex, instanceLabel, instanceWeight, value, indices, values))
  //        }
  //    }.join(model)
  //      .map { case (index, ((instanceIndex, instanceLabel, instanceWeight, value, indices, values), (weight, factor, selected))) =>
  //        (instanceIndex, (instanceLabel, instanceWeight, value, indices, values, index, weight, factor, selected))
  //      }.aggregateByKey((true, numH.zero, numH.zero, Array.empty[I], Array.empty[H], mutable.Set.empty[I], numH.zero, Array.ofDim(rank)[H], Array.ofDim(rank)[H]))(
  //      seqOp = {
  //        case ((isFirst, instanceLabel, instanceWeight, indices, values, selectedIndices, weightSum, dotSum, dot2Sum),
  //        (instanceLabel_, instanceWeight_, value_, indices_, values_, index_, weight_, factor_, selected_)) =>
  //
  //
  //
  //
  //
  //
  //
  //
  //          ???
  //      },
  //      combOp = {
  //        ???
  //      }
  //
  //    )
  //
  //
  //    ???
  //  }

}