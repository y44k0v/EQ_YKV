from __future__ import print_function
import numpy as np
import pyspark as ps
import math
import os
import urllib
import sys
import pyspark.sql.functions as F



from pyspark.sql import SparkSession

reload(sys)
sys.setdefaultencoding('utf-8')

spark = ps.sql.SparkSession.builder.appName('EQ_ykv').getOrCreate()

# Load data 

sample = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .load('gs://tester012018-192800/EQ/DataSample.csv')


poi = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .load('gs://tester012018-192800/EQ/POIList.csv')


# Stage 0 - CLEANUP

# Removing spaces from names
sample = sample.withColumnRenamed(" TimeSt", "TimeSt")
poi = poi.withColumnRenamed(" Latitude", "Latitude")

# data size

print("data size: \n")
print("sample : ", sample.columns)
print((sample.count(), len(sample.columns)))
print("pois : ", poi.columns)
print((poi.count(), len(poi.columns)))

# Temp tables
sample.registerTempTable("sample0")
poi.registerTempTable("pois0")

# duplicate removal 

query = """SELECT *
FROM sample0 A
WHERE _ID > (SELECT MIN(_ID) FROM sample0 B
WHERE A.TimeSt = B.TimeSt AND A.Latitude=B.Latitude AND A.Longitude=B.Longitude)"""

duplicates = spark.sql(query)
print("Duplicated data - time stamp and location : \n")
print(duplicates.count())

clean = sample.subtract(duplicates)

# Temp table
clean.registerTempTable("clean0")

# Removing mislabeled data out of canadian boundaries

query = """ SELECT *
FROM clean0
WHERE Latitude > 40 AND Longitude > -130 AND Longitude < -60
"""

clean2 = spark.sql(query)
print("Cleaned records: \n",clean2.count())
# Temp table
clean2.registerTempTable("clean2")

# 2 out of four POIs where the same / Location convertion to radians

poi = poi.withColumn("lat_rad", poi.Latitude* math.pi / 180)
poi = poi.withColumn("lon_rad", poi.Longitude* math.pi / 180)


# Saage 1 - minimum distance and POI Assignation

# Distance calculation and location conversion to radians

lat0 = F.toRadians("Latitude").alias("lat0")
lon0 = F.toRadians("Longitude").alias("lon0")
lat1 = 0.9345569159727344
lon1 = -1.9806997123424743
lat2 = 0.7945023069213337
lon2 = -1.2839693364011688
lat3 = 0.7893221871547071
lon3 = -1.1036193160713015
dlat1 = lat1 - lat0
dlon1 = lon1 - lon0
dlat2 = lat2 - lat0
dlon2 = lon2 - lon0
dlat3 = lat3 - lat0
dlon3 = lon3 - lon0
a1 = F.sin(dlat1/2)**2 + F.cos(lat0) * F.cos(lat0) * F.sin(dlon1/2)**2
a2 = F.sin(dlat2/2)**2 + F.cos(lat0) * F.cos(lat0) * F.sin(dlon2/2)**2
a3 = F.sin(dlat3/2)**2 + F.cos(lat0) * F.cos(lat0) * F.sin(dlon3/2)**2
c1 = F.lit(2) * F.asin(F.sqrt(a1)) 
c2 = F.lit(2) * F.asin(F.sqrt(a2)) 
c3 = F.lit(2) * F.asin(F.sqrt(a3)) 
r = F.lit(6371)
dist1 = (c1 * r).alias('dist1')
dist2 = (c2 * r).alias('dist2')
dist3 = (c3 * r).alias('dist3')

distances = clean2.select("_ID", "TimeSt", "City", "Province","Latitude", "Longitude", dist1, dist2, dist3)
distances.registerTempTable("dist0")

# POI assignation and minimal distance to poi

query = """SELECT _ID,  TimeSt, City, Province, dist1, dist2, dist3,
    CASE WHEN (dist1 < dist2) AND (dist1 < dist3) THEN "POI1 - EDMONTON" 
         WHEN (dist2 < dist1) AND (dist2 < dist3) THEN "POI2 - MONTREAL" 
         ELSE "POI3 - NOVA SCOTIA" 
         END AS POI
FROM dist0

"""

distPOI = spark.sql(query)
distPOI.registerTempTable("distPOI0")

query = """SELECT _ID, TimeSt, City,  POI,
    CASE WHEN (dist1 < dist2) AND (dist1 < dist3) THEN dist1 
         WHEN (dist2 < dist1) AND (dist2 < dist3) THEN dist2
         ELSE dist3 
         END AS minDist
FROM distPOI0
"""
distPOI2 = spark.sql(query)
distPOI2.registerTempTable("distPOI2")
distPOI2.show()


# Stage 2 Analysis

# grouping data by POI
by_POI = distPOI2.groupBy("POI")


by_POI.avg("minDist").show()

by_POI.agg(F.stddev("minDist")).show()

by_POI.min("minDist").show()

by_POI.max("minDist").show()

by_POI.agg(F.skewness("minDist")).show()

by_POI.agg(F.kurtosis("minDist")).show()


query = """SELECT COUNT(_ID) Requests, POI, AVG(minDist) AS Mean,  percentile_approx(minDist, 0.5) AS Median,
MAX(minDist) AS poiRadius_km, COUNT(_ID)/(3.14159*POWER(MAX(minDist),2)) AS Density_Requests_by_km2
FROM distPOI2
GROUP BY POI
"""
spark.sql(query).show()





