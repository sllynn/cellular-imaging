# Databricks notebook source
# MAGIC %md
# MAGIC # Automating Digital Pathology with <img src="https://databricks.com/wp-content/themes/databricks/assets/images/header_logo_2x.png" alt="logo" width="150"/> 
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/01/blog-digpath-1-1.png" alt="refarch" width="850" style="vertical-align:middle"/> 
# MAGIC 
# MAGIC The tumor proliferation speed or tumor growth is an important biomarker for predicting patient outcomes. Proper assessment of this biomarker is crucial for informing the decisions for the treatment plan for the patient. In a clinical setting, the most common method is to count [mitotic figures](https://www.mypathologyreport.ca/mitotic-figure/#:~:text=A%20mitotic%20figure%20is%20a,at%20tissue%20under%20the%20microscope) under a microscope by a pthologist. The manual counting and subjectivity of the process poses reproducibility challengs. This has been the main motivation for many efforts to automate this process and use advance ML techniques.
# MAGIC 
# MAGIC One of the main challenges however for autmating this task, is the fact that whole slide imgaes are rather large. WSI images can varry anywhere between 0.5 to 3.5GB in size, and that can slow down the image pre-processing step which is neccesary for any downstream ML application.
# MAGIC 
# MAGIC In this solution accelerator, we walk you through a step-by-step process to use databricks capabilities to perform image segmentation and pre-processing on WSI and train a binary classifier  that produces a metastasis probability map over a whole slide image (WSI).
# MAGIC 
# MAGIC The data used in this solution accelerator is from the [Camelyon16 Grand Challenge](http://gigadb.org/dataset/100439), along with annotations based on hand-drawn metastasis outlines. We also use curated annotations for this dataset obtained from Baidu Research [github repository](https://github.com/baidu-research/NCRF).
# MAGIC 
# MAGIC We use Apache Spark's paralelization capabilities to generate tumor/normal patches based on annotation data as well as feature extraction, using a pre-trained [InceptionV3](https://keras.io/api/applications/inceptionv3/)
# MAGIC 
# MAGIC This solution accelerator contains 5 notebooks (including this one):
# MAGIC - `0-setup`: Run the current notebook to create init scripts, and also download annotations and patch coordinations. We then store annotations in delta tables to be used in the next step for patch generation.
# MAGIC 
# MAGIC - `1-patch-generation`: This notebook generates patches from WSI and extracts features from each patch using InceptionV3 pre-trained model and stores all annoations and patches (for ~100K images) in delta.
# MAGIC 
# MAGIC - `2-model-training`: In this notebook we tune and train a binary classifier to calssify tumor/normal patches using features stored in delta from the previous step.
# MAGIC 
# MAGIC - `3-scoring`: This notebook we use the model trained in the previous step to generate a metastasis probability heatmap for a given slide.
# MAGIC 
# MAGIC - `definitions`: This notebook contains definitions for some of the functions that are used in multiple places (for example patch generation and pre processing)
# MAGIC 
# MAGIC For this Solution Accelerator we recommend using `8.1 ML ` runtime, using appropriate number of workers based on your usecase.

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Install <img src="https://openslide.org/images/openslide_logo.png">
# MAGIC 
# MAGIC 
# MAGIC For the first time running this workflow, you need to create an `Init Script` to install `openlide-tools` from [OpenSlide library](https://openslide.org/) on your cluster.
# MAGIC 
# MAGIC This script will need to be [attached to the cluster we are using](https://docs.databricks.com/user-guide/clusters/init-scripts.html#configure-a-cluster-scoped-init-script). In your cluster's `Init Script` configuration pane, add the path `/openslide/openslide-tools.sh`. 
# MAGIC After the cluster re-starts, go to Libraries tab on your cluster configuration and install `openslide-python` from PyPi.
# MAGIC 
# MAGIC After doing so, re-start your cluster and attached this notebook to the cluster before running the rest of the commands.

# COMMAND ----------

# DBTITLE 1,create init script
dbutils.fs.mkdirs('/openslide/')
dbutils.fs.rm('/openslide/openslide-tools.sh', True)
dbutils.fs.put('/openslide/openslide-tools.sh',
               """
               #!/bin/bash
               apt-get install -y openslide-tools
               """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Path specification and creation

# COMMAND ----------

WSI_PATH='/databricks-datasets/med-images/camelyon16/'
BASE_PATH='/tmp/digital-pathology'
ANNOTATION_PATH = BASE_PATH+"/annotations"
PATCH_PATH = BASE_PATH+"/patches"

# COMMAND ----------

import os
for path in [BASE_PATH, PATCH_PATH, ANNOTATION_PATH]:
  if not os.path.exists((f'/dbfs/{path}')):
    print(f"path {path} does not exist")
    dbutils.fs.mkdirs(path)
    print(f"created path {path}")
  else:
    print(f"path {path} exists")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Download annotations
# MAGIC We use pre-processed annotations from [BaiduResearch](https://github.com/baidu-research/NCRF). This repository, contains the coordinates of pre-sampled patches used in [the paper](https://openreview.net/forum?id=S1aY66iiM) which uses conditional random fields in conjunction with CNNs to achieve the highest accurarcy for detecting metastasis on WSI images:
# MAGIC 
# MAGIC >Each one is a csv file, where each line within the file is in the format like Tumor_024,25417,127565 that the last two numbers are (x, y) coordinates of the center of each patch at level 0. tumor_train.txt and normal_train.txt contains 200,000 coordinates respectively, and tumor_valid.txt and normal_valid.txt contains 20,000 coordinates respectively. Note that, coordinates of hard negative patches, typically around tissue boundary regions, are also included within normal_train.txt and normal_valid.txt. With the original WSI and pre-sampled coordinates, we can now generate image patches for training deep CNN models.
# MAGIC 
# MAGIC [see here](https://github.com/baidu-research/NCRF#patch-images) for more information

# COMMAND ----------

import subprocess
try:
   dbutils.fs.ls(f'{ANNOTATION_PATH}/tumor_train.txt')
except:
   subprocess.call('wget https://raw.githubusercontent.com/baidu-research/NCRF/master/coords/tumor_train.txt',shell=True,cwd=f'/dbfs{ANNOTATION_PATH}')
    
try:
  dbutils.fs.ls(f'{ANNOTATION_PATH}/normal_train.txt')
except:
  subprocess.call('wget https://raw.githubusercontent.com/baidu-research/NCRF/master/coords/normal_train.txt',shell=True,cwd=f'/dbfs{ANNOTATION_PATH}')

print(dbutils.fs.head(ANNOTATION_PATH+'/normal_train.txt', 111))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Quick view of files and annotations

# COMMAND ----------

print(f"WSI images are located in {WSI_PATH}, annotaions can be found in {ANNOTATION_PATH} and patches will be stored in {PATCH_PATH}")

# COMMAND ----------

# DBTITLE 1,Distribution of WSI sizes
display(dbutils.fs.ls(WSI_PATH))

# COMMAND ----------

# DBTITLE 1,Quick view of some of the slides
import numpy as np
import openslide
import matplotlib.pyplot as plt

f, axarr = plt.subplots(1,4,sharey=True)
i=0
for pid in ["normal_034","normal_036","tumor_044", "tumor_045"]:
  path = f'/dbfs/{WSI_PATH}/{pid}.tif'
  slide = openslide.OpenSlide(path)
  axarr[i].imshow(slide.get_thumbnail([m//50 for m in slide.dimensions]))
  axarr[i].set_title(pid)
  i+=1

# COMMAND ----------

# DBTITLE 1,Viewing slides at different zoom levels
slide = openslide.OpenSlide(f'/dbfs/{WSI_PATH}/normal_034.tif')
image_datas=[]
region=[35034,131012]
size=[4000,4000]
f, axarr = plt.subplots(2,3,sharex=True,sharey=True)
for level,ind in zip([0,1,2,3,4,5,6],[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]):
  img = slide.read_region(region,level,size)
  axarr[ind].imshow(img)
  axarr[ind].set_title(f"level:{level}")

display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create annotaion dataframes
# MAGIC Now we create a dataframe of tumor/normal coordinates based on the annotation data and write the result in delta tables to be used in the next stage for creating patches.

# COMMAND ----------

import pyspark.sql.functions as F

# load tumor patch coordinates and assign label = 0
df_coords_normal = spark.read.csv(f'{ANNOTATION_PATH}/normal_train.txt').withColumn('label', F.lit(0))

# load tumor patch coordinates and assign label = 1
df_coords_tumor = spark.read.csv(f'{ANNOTATION_PATH}/tumor_train.txt').withColumn('label', F.lit(1))

# union patches together
df_coords = df_coords_normal.union(df_coords_tumor).selectExpr('lower(_c0) as pid','_c1 as x_center', '_c2 as y_center', 'label')
df_coords.write.format('delta').mode('overWrite').save(f'{ANNOTATION_PATH}/delta/patch_labels')
display(df_coords)

# COMMAND ----------

