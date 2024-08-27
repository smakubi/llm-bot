# Databricks notebook source
# MAGIC %md 
# MAGIC ### CONFIG FILE
# MAGIC
# MAGIC In this config file you can change catalog and schema to run the workshop on a different catalog.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2Fconfig&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2Fconfig&version=1">

# COMMAND ----------

# HUGGING FACE TOKEN
hf_token = "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"

# COMMAND ----------

#GENERAL
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split(".")[0]
catalog = 'llm_workshop'
dbName = db = schema = f'{user_name}_rfp'
volume_name = 'rfp'
workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
base_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# COMMAND ----------

# CREATE STORAGE
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
spark.sql(f"USE SCHEMA {schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_name}") 

# COMMAND ----------

volume_folder = f"/Volumes/{catalog}/{db}/{volume_name}"

# COMMAND ----------

#EMBEDDING MODEL
embedding_model_name=f'{user_name}-gte-large-en'
registered_embedding_model_name = f'{catalog}.{schema}.{embedding_model_name}'
# registered_embedding_model_name = f'system.ai.{embedding_model_name}'
embedding_endpoint_name = f'{user_name}-gte-large-en' 

# COMMAND ----------

#VECTOR SEARCH
vs_endpoint_name="one-env-shared-endpoint-4" #change back
# vs_endpoint_name=f'{user_name}-sophinea-rfp-endpoint'
vs_index = f'{user_name}_sophinea_rfp_index'
vs_index_fullname = f"{catalog}.{schema}.{vs_index}"
sync_table_fullname = f"{catalog}.{schema}.sync_table_name"

# COMMAND ----------

#LLM SERVING
llm_model_name=f'{user_name}-llama-2-7b-hf-chat'
registered_llm_model_name=f'{catalog}.{schema}.{llm_model_name}'
llm_endpoint_name = f'{user_name}-llama-2-7b-hf-chat'