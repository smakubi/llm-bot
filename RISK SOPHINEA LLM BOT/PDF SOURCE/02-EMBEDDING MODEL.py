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
# MAGIC ## FUNCTIONS TO MANAGE THE MODEL ENDPOINT

# COMMAND ----------
# gather other inputs the API needs - they are used as environment variables in the
serving_host = spark.conf.get("spark.databricks.workspaceUrl")
creds = get_databricks_host_creds()

#
def endpoint_exists(serving_endpoint_name):
  """Check if an endpoint with the serving_endpoint_name exists"""
  url = f"https://{serving_host}/api/2.0/serving-endpoints/{serving_endpoint_name}"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  response = requests.get(url, headers=headers)
  return response.status_code == 200

# WAIT FOR ENDPOINT TO BE READY
def wait_for_endpoint(serving_endpoint_name):
  """Wait until deployment is ready, then return endpoint config"""
  headers = { 'Authorization': f'Bearer {creds.token}' }
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{serving_endpoint_name}"
  response = requests.request(method='GET', headers=headers, url=endpoint_url)
  while response.json()["state"]["ready"] == "NOT_READY" or response.json()["state"]["config_update"] == "IN_PROGRESS" : # if the endpoint isn't ready, or undergoing config update
    print("Waiting 30s for deployment or update to finish")
    time.sleep(30)
    response = requests.request(method='GET', headers=headers, url=endpoint_url)
    response.raise_for_status()
  return response.json()

# CREATE A SERVING ENDPOINT
def create_endpoint(serving_endpoint_name, served_models):
  """Create serving endpoint and wait for it to be ready"""
  print(f"Creating new serving endpoint: {serving_endpoint_name}")
  endpoint_url = f'https://{serving_host}/api/2.0/serving-endpoints'
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = {"name": serving_endpoint_name, "config": {"served_models": served_models}}
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.post(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint(serving_endpoint_name)
  displayHTML(f"""Created the <a href="/#mlflow/endpoints/{serving_endpoint_name}" target="_blank">{serving_endpoint_name}</a> serving endpoint""")
  
# UPDATE EXISTING ENDPOINT
def update_endpoint(serving_endpoint_name, served_models):
  """Update serving endpoint and wait for it to be ready"""
  print(f"Updating existing serving endpoint: {serving_endpoint_name}")
  endpoint_url = f"https://{serving_host}/api/2.0/serving-endpoints/{serving_endpoint_name}/config"
  headers = { 'Authorization': f'Bearer {creds.token}' }
  request_data = { "served_models": served_models, "traffic_config": traffic_config }
  json_bytes = json.dumps(request_data).encode('utf-8')
  response = requests.put(endpoint_url, data=json_bytes, headers=headers)
  response.raise_for_status()
  wait_for_endpoint(serving_endpoint_name)
  displayHTML(f"""Updated the <a href="/#mlflow/endpoints/{serving_endpoint_name}" target="_blank">{serving_endpoint_name}</a> serving endpoint""")

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
# MAGIC ## PREPARE DATA FOR QUERY

# COMMAND ----------

# Prepare data for query
# Query endpoint (once ready)
sentences = ['Hello world', 'Good morning']
ds_dict = {'dataframe_split': pd.DataFrame(pd.Series(sentences)).to_dict(orient='split')}
data_json = json.dumps(ds_dict, allow_nan=True)
print(data_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ## TEST ENDPOINT

# COMMAND ----------
# testing endpoint
if endpoint_exists(embedding_endpoint_name):
    invoke_headers = {'Authorization': f'Bearer {creds.token}', 'Content-Type': 'application/json'}
    invoke_url = f'{workspace_url}/serving-endpoints/{embedding_endpoint_name}/invocations'

    start = time.time()
    invoke_response = requests.request(method='POST', headers=invoke_headers, url=invoke_url, data=data_json, timeout=360)
    end = time.time()

    if invoke_response.status_code != 200:
        raise Exception(f'Request failed with status {invoke_response.status_code}, {invoke_response.text}')
    else:
        print(invoke_response.text)
else:
    wait_for_endpoint(embedding_endpoint_name)

# COMMAND ----------
