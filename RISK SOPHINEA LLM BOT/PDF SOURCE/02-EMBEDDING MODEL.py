# Databricks notebook source
# MAGIC %md
# MAGIC ## 02 - EMBEDDING MODEL
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The embedding model is what will be used to convert our **extracted PDF chunks into vector embeddings**. These vector embeddings will be loaded into a **Vector Search Index** and allow for fast **"Semantic Similarity Search"**. 
# MAGIC
# MAGIC This is a very important part of the RAGs architechture. We will be using the [GTE large](https://huggingface.co/thenlper/gte-large) embeddings model. This is an source embedding model you can get from hugging face or databricks unity catalog. 

# COMMAND ----------

# MAGIC %pip install --upgrade --quiet "mlflow-skinny[databricks]" sentence-transformers==2.6.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../RESOURCES/02-INIT-ADVANCED $reset_all_data=false 

# COMMAND ----------

import json
import pandas as pd
import requests
import time
from sentence_transformers import SentenceTransformer
from mlflow.utils.databricks_utils import get_databricks_host_creds
creds = get_databricks_host_creds()

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# DOWNLOAD EMBEDDING MODEL FROM HUGGING FACE
source_model_name = 'thenlper/gte-large'
model = SentenceTransformer(source_model_name)

# COMMAND ----------

# TEST THE MODEL JUST TO SHOW IT WORKS.
sentences = ["Checking if this works", "Each sentence is converted"]
embeddings = model.encode(sentences)
print(embeddings)

# COMMAND ----------

# COMPUTE INPUT/OUTPUT SCHEMA
signature = mlflow.models.signature.infer_signature(sentences, embeddings)
print(signature)

# COMMAND ----------

# START MLFLOW CLIENT
mlflow_client = mlflow.MlflowClient()

# COMMAND ----------

# REGISTER MODEL INTO UC
model_info = mlflow.sentence_transformers.log_model(
  model,
  artifact_path="model",
  signature=signature,
  input_example=sentences,
  registered_model_name=registered_embedding_model_name)

# WRITE A MODEL DESCRIPTION 
mlflow_client.update_registered_model(
  name=f"{registered_embedding_model_name}",
  description="gte from hugging face"
)

# COMMAND ----------

# GET THE LATEST VERSION OF THE MODEL
def get_latest_model_version(mlflow_client, model_name):
  model_version_infos = mlflow_client.search_model_versions("name = '%s'" % model_name)
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

model_version=get_latest_model_version(mlflow_client, registered_embedding_model_name)
print(model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## CREATING MODEL SERVING ENDPOINT

# COMMAND ----------

# SERVE EMBEDDING MODEL ON A MODEL SERVING ENDPOINT (30 MINS)
deploy_headers = {'Authorization': f'Bearer {creds.token}', 'Content-Type': 'application/json'}
deploy_url = f'{workspace_url}/api/2.0/serving-endpoints'
endpoint_config = {
  "name": embedding_endpoint_name,
  "config": {
    "served_models": [{
      "name": f'{embedding_model_name}',
      "model_name": registered_embedding_model_name,
      "model_version": model_version,
      "workload_type": "GPU_MEDIUM", # NEED FASTER SERVING FOR VECTOR SEARCH
      "workload_size": "Medium", # CHANGE TO MEDIUM
      "scale_to_zero_enabled": False,
    }]
  }
}

endpoint_json = json.dumps(endpoint_config, indent='  ')
deploy_response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)
if deploy_response.status_code != 200:
  raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')

print(deploy_response.json())

# COMMAND ----------

# MAGIC %md
# MAGIC ## DELETING THE ENDPOINT

# COMMAND ----------

# # Assuming you have the endpoint ID and credentials
# endpoint_id = deploy_response.json()['id']
# print(endpoint_id)
# delete_headers = {'Authorization': f'Bearer {creds.token}', 'Content-Type': 'application/json'}
# delete_url = f'{workspace_url}/api/2.0/serving-endpoints/{endpoint_id}'

# delete_response = requests.request(method='POST', headers=delete_headers, url=delete_url)

# if delete_response.status_code != 200:
#   raise Exception(f'Delete failed with status {delete_response.status_code}, {delete_response.text}')

# print("Endpoint deleted successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## OR: MLFLOW DEPLOYMENT SDK

# COMMAND ----------

# from mlflow.deployments import get_deploy_client

# client = get_deploy_client("databricks")
# client.delete_endpoint(endpoint=embedding_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## UPDATING THE ENDPOINT

# COMMAND ----------

# # Assuming you have the endpoint ID and credentials
# endpoint_id = deploy_response.json()['id']
# update_headers = {'Authorization': f'Bearer {creds.token}', 'Content-Type': 'application/json'}
# update_url = f'{workspace_url}/api/2.0/serving-endpoints/{endpoint_id}'

# # Configuration to update the endpoint
# update_config = {
#   "config": {
#     "scale_to_zero_enabled": True
#   }
# }
# update_response = requests.request(method='POST', headers=update_headers, url=update_url, data=json.dumps(update_config))

# if update_response.status_code != 200:
#   raise Exception(f'Update failed with status {update_response.status_code}, {update_response.text}')

# print("Endpoint updated successfully:", update_response.json())

# COMMAND ----------

# MAGIC %md
# MAGIC ## PREPARE DATA FOR QUERY

# COMMAND ----------

# Prepare data for query
# Query endpoint (once ready)
sentences = ['Hello world', 'Good morning']
ds_dict = {'dataframe_split': pd.DataFrame(pd.Series(sentences)).to_dict(orient='split')}
data_json = json.dumps(ds_dict, allow_nan=True)
print(data_json)

# COMMAND ----------

# testing endpoint
invoke_headers = {'Authorization': f'Bearer {creds.token}', 'Content-Type': 'application/json'}
invoke_url = f'{workspace_url}/serving-endpoints/{embedding_endpoint_name}/invocations'
print(invoke_url)

start = time.time()
invoke_response = requests.request(method='POST', headers=invoke_headers, url=invoke_url, data=data_json, timeout=360)
end = time.time()
print(f'time in seconds: {end-start}')

if invoke_response.status_code != 200:
  raise Exception(f'Request failed with status {invoke_response.status_code}, {invoke_response.text}')

print(invoke_response.text)