package com.app
import org.apache.spark.sql.functions._
import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.io.PrintWriter
import java.io.File
import org.apache.spark.sql.types._
import org.apache.commons.lang3.time.DateUtils
import org.apache.commons.math3.stat.StatUtils
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics
import org.apache.spark.sql._
import java.sql.Timestamp
import java.sql.Date
import java.util.Date;
import org.apache.spark.sql.types._
import org.apache.spark._
import org.apache.commons.lang3.time.DateUtils
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row


import org.apache.spark.{SparkConf, SparkContext}

object getFeature3 {
  //init the spark
  val conf = new SparkConf()
    .setMaster("yarn-cluster")
    .setAppName("feature data")
    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  val sc = new SparkContext(conf)
  sc.setLogLevel("WARN")
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  val fmt: SimpleDateFormat = new java.text.SimpleDateFormat("yyyy/MM/dd")

  def main(args: Array[String]):Unit = {
    /* *
    *args(0): startDay,"2017/07/ -/"
    *val rootpath = "/datalab/user/frank.zhang/data/"
    *val watchDay = "2017/07/01"
    * val period = 2
    * val appKey = "323BE90BCA2213A07D18FE935A6BA9E5"

    *读取app名称
    *[52F459D41AAB85846D4CA031B40FFECE,360手机助手]
    *[C365D790EDFCDC1C734F1272496C87D5,墨迹天气iOS]
    *[323BE90BCA2213A07D18FE935A6BA9E5,墨迹天气Android]
    *[86C480859A2547CB85FB0BC5A6EC3943,秒拍]
    */

    import sqlContext.implicits._
    val rootpath = args(0)
    val watchDay = args(1)
    val period = args(2).toInt
    val numTree = args(3).toInt
    val treeDepth = args(4).toInt

    val appKey = "323BE90BCA2213A07D18FE935A6BA9E5"
    val appdf =loadPeriodData(rootpath,watchDay,period,appKey)
    val datadf = getFeature3.getActionFeatures(appdf,period,2.0,watchDay).repartition(6)

    sc.parallelize(Array(datadf.count)).repartition(1).saveAsTextFile("/datalab/user/frank.zhang/data/result/tmp"+System.currentTimeMillis().toString)
    val outpath = "/datalab/user/frank.zhang/data/feature/action/"+watchDay+"-14"
    saveData(datadf,outpath)
    // 2-training phase
    val actionPath = outpath
    val labelperiod = 7
    val training = getPredict.getTrainData(actionPath,rootpath,watchDay,labelperiod,appKey,0.5)
    getPredict.RFModel(training,numTree,treeDepth)
  }
  def deleteFile(path: String) : Unit = {
    // delete files
    import org.apache.hadoop.conf.Configuration
    import org.apache.hadoop.fs.{FileSystem, Path}
    val hdfs : FileSystem = FileSystem.get(new Configuration)
    val isExist = hdfs.exists(new Path(path))
    if (isExist){
      println("welll")
      hdfs.delete(new Path(path), true)//true: delete files recursively
    }
  }
  /**seve the DataFrame to the specified path
    *detect the path is exists when exists delete it. then save the data
    */
  def saveData(df:org.apache.spark.sql.DataFrame,path:String):Unit ={
    import org.apache.hadoop.conf.Configuration
    import org.apache.hadoop.fs.{FileSystem, Path}
    val hdfs : FileSystem = FileSystem.get(new Configuration)
    val isExist = hdfs.exists(new Path(path))
    if (isExist){
      hdfs.delete(new Path(path), true)//true: delete files recursively
    }

    df.write.format("parquet").save(path)
  }
  // load DataFrame format data from paequet files
  def loadData(readPath: String):org.apache.spark.sql.DataFrame ={
    val df = sqlContext.read.parquet(readPath)
    df
  }
  def loadPeriodData(rootpath : String,watchDay : String,period : Integer,appKey : String) = {
    /**
      * val rootpath = "/datalab/user/frank.zhang/data/"
      * val  watchDay = "2017/09/01"
      */


    import sqlContext.implicits._
    // val rootpath = "/datalab/user/frank.zhang/data/"
    // val watchDay = "2017/09/01"

    val baseday = fmt.parse(watchDay)
    val beforeDay = DateUtils.addDays(baseday,-1)
    val tmpPath = rootpath + fmt.format(beforeDay)
    var perioddata = sqlContext.read.parquet(tmpPath).filter("appKey = '%s' ".format(appKey)) // variable!!! cost 1 afternoon and 1 night.
    // println(perioddata.count)
    for (iday <- -period to -2){
      val day = DateUtils.addDays(baseday,iday)
      val dayStr = fmt.format(day)
      val dayPath = rootpath + dayStr
      val dayData = sqlContext.read.parquet(dayPath).filter("appKey = '%s' ".format(appKey))
      // println(dayPath)
      // println(dayData.count)
      // dayData.show(3)
      perioddata = perioddata.unionAll(dayData)
      // println(perioddata.count)

    }
    perioddata
  }
  /*
  * get features of month data*/
  def getActionFeatures(df:org.apache.spark.sql.DataFrame,period: Integer,sumBound: Double,watchDay:String) = {
    // 30 days data
    //Need :[tdid: string, appKey: string, receiveTime: date, platform: int, installTime: date, purchaseTime: date, brand: string, osStandardVersion: string, freq: int]
    import sqlContext.implicits._
    import org.apache.spark.util.StatCounter
    import scala.util.matching.Regex
    val getOSVer =udf[Double,String]{str =>
      // convert os string to double
      val pattern = new Regex("\\d")
      val version = (pattern findFirstIn str).mkString
      // val res = version.toDouble
      val res = if (version.length == 1) version.toDouble  else 5.0
      res
    }


    val monthDF =df.rdd.map{row =>
      val tdid = row.getString(0);
      val receiveTime = row.getDate(2)
      val installTime = row.getDate(4)
      val osStandardVersion = row.getString(7)
      val standardModel = row.getString(6)
      val freq =row.getInt(8);
      ((tdid,receiveTime,installTime,standardModel),freq)}
      .reduceByKey(_ + _).map(row =>(row._1._1,row._1._2,row._1._3,row._1._4,row._2))
      .toDF("tdid","receiveTime","installTime","standardModel","freq")

    val dataRDD = monthDF.select("tdid","receiveTime","freq").map{row=>
      val tdid= row.getAs[String]("tdid")
      val receiveTime = row.getAs[java.sql.Date]("receiveTime")
      val freq=row.getAs[Int]("freq").toDouble
      (tdid,(receiveTime.getTime(),freq))

    }
    type NewType=(Long,Double)
    //合并在同一个partition中的值，a的数据类型为zeroValue的数据类型，b的数据类型为原value的数据类型
    def seqOp = (u : List[NewType],v : NewType)=>{
      u:+v
    }
    //合并不同partition中的值，a，b得数据类型为zeroValue的数据类型
    def combOp(a:List[NewType],b:List[NewType])={
      a:::b.tail
    }
    //zeroValue:中立值,定义返回value的类型，并参与运算
    //seqOp:用来在同一个partition中合并值
    //combOp:用来在不同partiton中合并值
    val dayms=1000*3600*24.0
    val aggregateByKeyRDD=dataRDD.aggregateByKey(List[NewType]((fmt.parse(watchDay).getTime(),0.0)))(seqOp, combOp)
    // time_diff
    def udfMinus(x :NewType,y:NewType): Double=(x._1-y._1)/dayms
    val aggregatedDF=aggregateByKeyRDD.map{row=>

      val sortedByTime = row._2.sortBy(tupe=>tupe._1)
      val timeRowAll=(0 to row._2.length-2).map((n:Int)=> udfMinus(sortedByTime.tail(n),sortedByTime.init(n))).toArray
      val timeRow = timeRowAll.init
      val timeCounter = new StatCounter((0 to row._2.length-2).map((n:Int)=> udfMinus(sortedByTime.tail(n),sortedByTime.init(n))).init.toTraversable) //need change

      val freqRow = (0 to sortedByTime.length-2).map((i:Int)=>sortedByTime(i)._2).toArray
      val freqCounter = new StatCounter(freqRow.toTraversable)

      val tNearUseDiff=timeRowAll.last
      val tNearUseDay = new java.sql.Date(sortedByTime(sortedByTime.length-2)._1)
      val tMean= timeCounter.mean;
      val tMax= timeCounter.max;
      val tMin= timeCounter.min;
      val tVar= timeCounter.variance;
      val fMean= freqCounter.mean;
      val fMax= freqCounter.max;
      val fMin= freqCounter.min;
      val fVar= freqCounter.variance;
      val fSum= StatUtils.sum(freqRow);

      val fPeriodActive = freqCounter.count;
      val fDayActive= StatUtils.sum(freqRow)/period;
      (row._1,tMean,tMax,tMin,tVar,fMean,fMax,fMin,fVar,fSum,fPeriodActive,fDayActive,tNearUseDiff,tNearUseDay)
    }.toDF("tdid","tMean","tMax","tMin","tVar","fMean","fMax","fMin","fVar","fSum","fPeriodActive","fDayActive","tNearUseDiff","tNearUseDay").filter("fPeriodActive > %s".format(sumBound))

    val selectDF = monthDF.select("tdid","installTime","standardModel").dropDuplicates()
    val featureTmpDF=aggregatedDF.join(selectDF,Seq("tdid"),"left").dropDuplicates()
    val featureDF = featureTmpDF.withColumn("tAliveTime",datediff($"tNearUseDay",$"installTime").cast(DoubleType)).drop("tNearUseDay").drop("installTime").filter("tAliveTime >%s".format(7))
    //tdid: string, appKey: string, receiveTime: date, platform: int, installTime: date, purchaseTime: date, brand: string, osStandardVersion: string, payAmount: double, freq: bigint, tMean: double, tMax: double, tMin: double, tVar: double, fMean: double, fMax: double, fMin: double, fVar: double, fSum: double, fdayMean: double, tNearUseDiff: double, tNearUseDay: date, tAliveTime: int
    // mergeFeagure.select("tdid","osStandardVersion","tMean","tMax","tMin","tVar","fMean","fMax","fMin","fVar","fSum","fPeriodActive","fDayActive","tNearUseDiff","tAliveTime")
    featureDF
  }
  def mergeFeatures(actionPath : String,appPath : String,savePath : String) : org.apache.spark.sql.DataFrame = {
    /**
      * Merge the features (action,device,app)
      * deal with missing values
      */
    import sqlContext.implicits._
    val actionInfo = sqlContext.read.parquet(actionPath).drop("fMin").drop("tMin")
    // rawDF: org.apache.spark.sql.DataFrame = [tdid: string, appKey: string, receiveTime: date, platform: int, installTime: date, purchaseTime: date, brand: string, osStandardVersion: string, payAmount: double, freq: bigint]
    val devicePath = "/datalab/user/frank.zhang/data/deviceinfo"
    val deviceInfo = sqlContext.read.parquet(devicePath)
    val appInfo = getPredict.loadData(sqlContext,appPath)
    // val merge1 =  actionInfo.join(deviceInfo,$"standardModel" === $"standard_model","left").drop("standardModel").drop("standard_model").distinct()
    val mergeDF1 = actionInfo.join(appInfo,Seq("tdid"),"left").na.drop()
    //[standard_model: string, price_range: double, hardware_type: double, network_type: double, RAM: string]
    val mergeDF2 = mergeDF1.join(deviceInfo,$"standardModel" === $"standard_model","left").na.drop
    // val featDF = getFeatures2(monthDF,30.0,1.0,"2017/05/01")
    //featDF:("tdid","tMean","tMax","tMin","tVar","fMean","fMax","fMin","fVar","fSum","fdayMean","nearUseDiff")
    // val initInstallPath ="/datalab/user/frank.zhang/data/feature/initapp2dfreq"
    // val initInstallDF = sqlContext.read.parquet(initInstallPath)
    // val featAll = merge1.join(initInstallDF,Seq("tdid"),"left").na.fill(2.0,Array("initUsefreq"))
    val toDouble = udf[Double, String]( _.toDouble)
    // val featAllDF = featAll.withColumn("initUsefreq",toDouble($"initUsefreq"))
    // featAllDF.write.format("parquet").save(savePath)
    mergeDF2

  }
}
