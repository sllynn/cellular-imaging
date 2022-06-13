# Databricks notebook source
# MAGIC %md
# MAGIC # Functions and Definitions
# MAGIC This notebook contains definitions for functions that will be used multiple times in this solution accelerator

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. distributed patch extraction
# MAGIC <img src="https://openslide.org/images/openslide_logo.png">
# MAGIC 
# MAGIC Below we define two functions: `get_patch` and `process_patch`. `get_patch` takes the name of the slide, x and y coordinates of the center of the patch to be extarcted and the label for the given patch (0 for normal and 1 for tumor). It then returns the content of the extracted patch as a numeric array (flattened numpy array in as a spark vector).  `process_patch` is a pandas_udf that distributes this function on the annotations dataframe to obtain a dataframe of processed patches. 
# MAGIC 
# MAGIC To extract patches and manipulate WSI images, we use the [OpenSlide library](https://openslide.org/), which is assumed to be installed during the clsuter configuration using [init script](https://docs.databricks.com/user-guide/clusters/init-scripts.html)

# COMMAND ----------

from typing import Iterator
import pandas as pd

import openslide
from tensorflow.keras.applications.inception_v3 import preprocess_input 

# COMMAND ----------

# from pyspark.ml.linalg import VectorUDT, Vectors
def get_patch(pid,x_center,y_center):
  path=f'/dbfs{WSI_PATH}{pid}.tif'
  path=path.replace('dbfs:','/dbfs')
  slide = openslide.OpenSlide(path)
  x = int(x_center) - PATCH_SIZE // 2
  y = int(y_center) - PATCH_SIZE // 2
  img = slide.read_region((x, y), LEVEL,(PATCH_SIZE, PATCH_SIZE)).convert('RGB')
  img_arr=np.asarray(img)
  processed_img_arr=preprocess_input(img_arr).flatten()
  return (processed_img_arr)

def process_patch (pdf_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
  for pdf in pdf_iter:
    _pdf=pdf[['pid','x_center','y_center','label']]
    processed_img_pdf=pd.DataFrame({
      'label':_pdf['label'],
      'x_center' : _pdf['x_center'],
      'y_center' : _pdf['y_center'],
      'processed_img':_pdf.apply(lambda x:get_patch(x['pid'],x['x_center'],x['y_center']),axis=1)
    })
    yield processed_img_pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. distributed feature extraction
# MAGIC 
# MAGIC <img src="https://cloud.google.com/tpu/docs/images/inceptionv3onc--oview.png">
# MAGIC 
# MAGIC We will use a pre-trained deep neural network (InceptionV3) to extract features from each patch.
# MAGIC 
# MAGIC 1. `model_fn`:  Returns a [InceptionV3](https://arxiv.org/abs/1512.00567) model with top layer removed and broadcasted pretrained weights.
# MAGIC 2. `featurize_series`: Featurize a pd.Series of raw images using the input model and return pd.Series of image features
# MAGIC 3. `featurize_pudf`: This method is a Scalar Iterator pandas UDF wrapping our featurization function. The decorator specifies that this returns a Spark DataFrame column of type `ArrayType(FloatType)`.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np

# COMMAND ----------

# loading Inception model
model = InceptionV3(include_top=False)
broadcaseted_model_weights = sc.broadcast(model.get_weights())

def model_fn():
  """
  Returns a InceptionV3 model with top layer removed and broadcasted pretrained weights.
  """
  model = InceptionV3(weights=None, include_top=False)
  model.set_weights(broadcaseted_model_weights.value)
  return model

def featurize_series(model, content_series):
  """
  Featurize a pd.Series of raw images using the input model.
  :return: a pd.Series of image features
  """
  _input = np.stack(content_series.map(lambda a: np.array(a).reshape(PATCH_SIZE,PATCH_SIZE,3)))
  preds = model.predict(_input)
  # For some layers, output features will be multi-dimensional tensors.
  # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
  output = [p.flatten() for p in preds]
  return pd.Series(output)


@pandas_udf('array<float>')
def featurize_pudf (content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
  '''
  This method is a Scalar Iterator pandas UDF wrapping our featurization function.
  The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).
  
  :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
  '''
  # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
  # for multiple data batches.  This amortizes the overhead of loading big models.
  model = model_fn()
  for content_series in content_series_iter:
    yield featurize_series(model, content_series)