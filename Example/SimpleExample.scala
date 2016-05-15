/**
 * Created by yuhao on 5/14/16.
 */
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.SVM
import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

object SimpleExample {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val spark = SparkSession.builder.appName("svm").master("local[8]").getOrCreate()

    val data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

    val model = new SVM().setKernelType("linear").fit(data)
    val result = model.transform(data).cache()
    result.show()
    println("total: " + data.count())
    println(result.filter("label = prediction").count())
  }

}
