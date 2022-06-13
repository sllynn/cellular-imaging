# Databricks notebook source
# MAGIC %md
# MAGIC # Distributed patch generation and pre-processing
# MAGIC In this notebook we use spark's `pandas_udfs` to effiently distribute patch generation process. 

# COMMAND ----------

WSI_PATH='/databricks-datasets/med-images/camelyon16/'
BASE_PATH='/tmp/digital-pathology'
ANNOTATION_PATH = BASE_PATH+"/annotations"
PATCH_PATH = BASE_PATH+"/patches"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Load Definitions

# COMMAND ----------

# MAGIC %run ./definitions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load pre-processed annotations
# MAGIC Now we load annotations generated in the previous step

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

# DBTITLE 1,Load annotations
coordinates_df = spark.read.load(f'{ANNOTATION_PATH}/delta/patch_labels')

df_patch_info = (
  spark.createDataFrame(dbutils.fs.ls(WSI_PATH))
  .withColumn('pid',lower(regexp_replace('name', '.tif', '')))
  .join(coordinates_df, on='pid')
  .select('pid','x_center','y_center','label')
)
print(f'there are {df_patch_info.count()} patches to process.')
display(df_patch_info)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create patches from WSI images
# MAGIC 
# MAGIC In this step, we simply use a pre-trained deep neural network (InceptionV3) to extract features from each patch. In the next step (`2-model-training`) we use these features to perform a binary classification using spark ml. The input to the network is a `3X3X299` tensor and the output is the output of the last pooling layer of the network with shape `8X8X2048`.
# MAGIC 
# MAGIC `Note`: This is for demonstration purposes and in real applications for training you can fine-trune models such as inception. It is often enough to use tensorflow or pytorch for distribution of the training tasks on a single node (with multiple GPUs). 
# MAGIC 
# MAGIC Since we are using inceptionV3, we use patch size of `299`. `LEVEL` corresponds to the zoom level (`0` being highest resolution).
# MAGIC 
# MAGIC We then apply feature extraction functions (`process_patch` and `featurize_pudf`) to our `patch_info` dataframe and write the resultant dataframe to delta for the next step. Note that the advantage of storing the intermediate patch information in delta is that you can combine different patches from different experiments together. You can later use these datasets for other purposes such as unsupervised ML to explore patterns in your data.

# COMMAND ----------

from pyspark.ml.linalg import VectorUDT, Vectors

# COMMAND ----------

PATCH_SIZE=299
LEVEL=0

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

# COMMAND ----------

# DBTITLE 1,Create a dataframe of processed patches
dataset_df = (
  df_patch_info.repartition(1024)
  .mapInPandas(process_patch, schema='processed_img:array<float>, label:integer')
  .select(featurize_pudf("processed_img").alias('features'),'label')
  .select('label',udf(lambda x: Vectors.dense(x), VectorUDT())('features').alias('features_vec'))
)

# COMMAND ----------

dataset_df.write.format('delta').mode('overwrite').save(f'{BASE_PATH}/delta')

# COMMAND ----------

spark.sql(f"OPTIMIZE delta.`{BASE_PATH}/delta`")

# COMMAND ----------

display(dbutils.fs.ls(f"{BASE_PATH}/delta"))