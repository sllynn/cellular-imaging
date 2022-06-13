# Databricks notebook source
# MAGIC %md
# MAGIC # Prediction on a new slide
# MAGIC Now let's use the trained model to identify regions that might contain metastatic sites.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Load function definition for patchin

# COMMAND ----------

# MAGIC %run ./definitions

# COMMAND ----------

WSI_PATH='/databricks-datasets/med-images/camelyon16/'
BASE_PATH='/tmp/digital-pathology'
ANNOTATION_PATH = BASE_PATH+"/annotations"
PATCH_PATH = BASE_PATH+"/patches"
PATCH_DELTA_PATH = BASE_PATH+"/delta"

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.linalg import VectorUDT, Vectors

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Image segmentation
# MAGIC To visualize the heatmap of probability of metastasis on a segment of the slide, first we need to create a grid of patches and then for each patch we run prediction based on the model that we trained in the previous step. to do so we leverage pathcning and pre-processing functions that we used in pre procesing step for training the mode. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Grid generation
# MAGIC The following function, takes the `x`,`y` coordinates of the boundries of the segment of a given slide for scroing and outputs a dataframe containing coordinates of each segment (segments of size `299X299`) and `i,j` indices corresponding to each segment within the grid.

# COMMAND ----------

def generate_patch_grid_df(*args):
  x_min,x_max,y_min,y_max, slide_name = args
  
  x = np.array(range(x_min,x_max,PATCH_SIZE))
  y = np.array(range(y_min,y_max,PATCH_SIZE))
  xv, yv = np.meshgrid(x, y, indexing='ij')
  
  cSchema = StructType([
    StructField("x_center", IntegerType()),
    StructField("y_center", IntegerType()),
    StructField("i_j", StringType()),
  ])

  arr=[]
  for i in range(len(x)):
    for j in range(len(y)):
      x_center = int(xv[i,j].astype('int'))
      y_center = int(yv[i,j].astype('int'))
      arr+=[[x_center,y_center,"%d_%d"%(i,j)]]
  grid_size = xv.shape
  df = spark.createDataFrame(arr,schema=cSchema) 
  return(df,grid_size)

# COMMAND ----------

# MAGIC %md
# MAGIC Next we generate patches based on a generated grid over a pre-selected segment of the slide.

# COMMAND ----------

PATCH_SIZE=299
LEVEL=0
name="tumor_058"
x_min,x_max,y_min,y_max = (23437,53337,135815,165715)

# COMMAND ----------

df_patch_info,grid_size = generate_patch_grid_df(x_min,x_max,y_min,y_max,name)
df_patch_info=df_patch_info.selectExpr(f"'{name}' as pid","x_center","y_center","i_j as label")
display(df_patch_info)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Patch generation
# MAGIC Now we apply patch pre-processing to the grid dataframe to generate a dataframe of features that will be fed to the pre-trained classifier for prediction.

# COMMAND ----------

dataset_df = (
  df_patch_info.repartition(1024)
  .mapInPandas(process_patch, schema='label:string, x_center: integer, y_center: integer, processed_img:array<float>')
  .select(featurize_pudf("processed_img").alias('features'),'label','x_center','y_center')
  .select('x_center','y_center','label',udf(lambda x: Vectors.dense(x), VectorUDT())('features').alias('features_vec'))
  .cache()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Scoring
# MAGIC Now that we have the dataframe of segments, we simply load our classifer using the `uri` returned in the previous notebook and load the model. Next we use this model for prediction on the input spark dataframe in parallel.

# COMMAND ----------

import mlflow
# this URI will need to be updated with your MLFlow run
model_uri='dbfs:/databricks/mlflow-tracking/9827568/b1432209f8914776bb5609db6b3b1b8c/artifacts/wsi-spark-lr'
model = mlflow.spark.load_model(model_uri=model_uri)

# COMMAND ----------

get_p_udf=udf(lambda v:float(v[1]),FloatType())
predictions_df = model.transform(dataset_df).withColumn('p',get_p_udf('probability')).drop('features_vec','probability','rawPrediction').cache()
predictions_pdf=predictions_df.toPandas()
display(predictions_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Visualization
# MAGIC Now that we have the probability scores for each segment along with the indices of each segment on the grid, we can create a simple heatmap of probabiliy scores.

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC Here is how the original slide and the selected segment look like:

# COMMAND ----------

# DBTITLE 1,selected slide segment
pid="tumor_058"
slide = openslide.OpenSlide('/dbfs/%s/%s.tif' %(WSI_PATH,pid))
region= [x_min,y_min]

size=[2900,2900]
slide_segment= slide.read_region(region,3,size)

f, axarr = plt.subplots(1,2)
axarr[0].imshow(slide_segment)
axarr[0].set_xlim=3000
axarr[0].set_ylim=3000
axarr[1].imshow(slide.get_thumbnail(np.array(slide.dimensions)//50))
axarr[1].axis('off')
f.set_figheight(12)
f.set_figwidth(12)
display()

# COMMAND ----------

# MAGIC %md
# MAGIC And here is the heatmap of probability scores:

# COMMAND ----------

# DBTITLE 1,metastatic heatmap
x_min,x_max=predictions_pdf['x_center'].min(),predictions_pdf['x_center'].max()
y_min,y_max=predictions_pdf['y_center'].min(),predictions_pdf['y_center'].max()
pred_arr=predictions_pdf[['label','p']]
n_x,n_y=grid_size
width,height=299,299
scale_f=0.2

img_size = int(scale_f*width),int(scale_f*height)
total_width = img_size[0]*n_x
total_height = img_size[1]*n_y

y, x = np.meshgrid(np.linspace(x_min, x_max, n_x), np.linspace(y_min, y_max, n_y))
z=np.zeros(y.shape)

for ij,p in pred_arr.values:
      i = int(ij.split('_')[0])
      j = int(ij.split('_')[1])
      z[i][j]=p

z = z[:-1, :-1]
z_min, z_max = -np.abs(z).max(), np.abs(z).max()

fig, ax = plt.subplots()

ax.imshow(slide_segment)
c = ax.pcolormesh(x, y, z, cmap='magma', vmin=z_min, vmax=z_max)
ax.set_title('metastasis heatmap')

# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
fig.set_figheight(12)
fig.set_figwidth(12)
plt.show()

# COMMAND ----------

