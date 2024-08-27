# Databricks notebook source
# MAGIC %md
# MAGIC ### INSTALL EXTERNAL LIBRARIES

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade transformers==4.41.1 pypdf==4.1.0 langchain-text-splitters==0.2.0 databricks-vectorsearch mlflow tiktoken==0.7.0 torch==2.3.0 llama-index==0.10.43
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### LOAD VARIABLES

# COMMAND ----------

# MAGIC %run ../RESOURCES/02-INIT-ADVANCED $reset_all_data=false 

# COMMAND ----------

# MAGIC %md
# MAGIC ## CREATE VECTOR SEARCH ENDPOINT

# COMMAND ----------

# IMPORT VECTOR SEARCH CLASS AND INITIATE A CLIENT
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# COMMAND ----------

# CREATE THE VECTOR SEARCH ENDPOINT. THIS WILL TAKE 15 MINS
if not endpoint_exists(vsc, vs_endpoint_name):
    vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")
    wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)
else:
    print(f"Endpoint named {vs_endpoint_name} is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## NOW CREATE VECTOR SEARCH INDEX

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### WHAT'S REQUIRED FOR OUR VECTOR SEARCH INDEX
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-vector-search-type.png?raw=true" style="float: center" width="800px">
# MAGIC
# MAGIC Databricks provide multiple types of vector search indexes:
# MAGIC
# MAGIC - **Managed embeddings**: you provide a text column and endpoint name and Databricks synchronizes the index with your Delta table 
# MAGIC - **Self Managed embeddings**: you compute the embeddings and save them as a field of your Delta Table, Databricks will then synchronize the index
# MAGIC - **Direct index**: when you want to use and update the index without having a Delta Table.
# MAGIC
# MAGIC In this workshop, we will show you how to setup a **Self-managed Embeddings** index. 
# MAGIC
# MAGIC To do so, we will have to first compute the embeddings of our chunks and save them as a Delta Lake table field as `array&ltfloat&gt`

# COMMAND ----------

# MAGIC %md
# MAGIC ### WE'LL USE GTE EMBEDDINGS MODEL

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-4.png?raw=true" style="float: center; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### TEST GTE EMBEDDINGS MODEL

# COMMAND ----------

from mlflow.utils.databricks_utils import get_databricks_host_creds
creds = get_databricks_host_creds()

# COMMAND ----------

from mlflow.deployments import get_deploy_client
from pprint import pprint

deploy_client = get_deploy_client("databricks")
embeddings = deploy_client.predict(
    endpoint=embedding_endpoint_name,
    inputs={"inputs": ["Checking if this works", "Each sentence is converted"]},
)
print(embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ### FINAL SOPHINEA PDF TABLE TO HOST EMBEDDINGS

# COMMAND ----------

# MAGIC %sql
# MAGIC -- WE NEED TO ENABLE CHANGE DATA FEED ON THE TABLE TO CREATE THE INDEX 
# MAGIC CREATE TABLE IF NOT EXISTS sophinea_pdf_documentation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY < FLOAT >
# MAGIC )
# MAGIC TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# MAGIC %md
# MAGIC ### COMPUTING THE CHUNK EMBEDDINGS AND SAVING TO DELTA TABLE 
# MAGIC
# MAGIC The last step is to now compute an embedding for all our documentation chunks. Let's create an udf to compute the embeddings using the embedding model endpoint.
# MAGIC
# MAGIC *Note that this part would typically be setup as a production-grade job, running as soon as a new documentation page is updated. <br/> This could be setup as a **Delta Live Table pipeline to incrementally consume updates**.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### UDF TO COMPUTE CHUNK EMBEDDINGS

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    deploy_client = get_deploy_client("databricks")

    def get_embeddings(batch):
        response = deploy_client.predict(
            endpoint=embedding_endpoint_name,
            inputs={"inputs": batch}
        )
        return response['predictions']

    # EMBEDDING MODEL TAKES AT MOST 150 INPUTS PER REQUEST
    max_batch_size = 150
    
    all_embeddings = []
    for i in range(0, len(contents), max_batch_size):
        batch = contents.iloc[i:i+max_batch_size].tolist()
        embeddings = get_embeddings(batch)
        all_embeddings.extend(embeddings)

    # Ensure we return the same number of embeddings as input rows
    assert len(all_embeddings) == len(contents), f"Mismatch in number of embeddings: got {len(all_embeddings)}, expected {len(contents)}"
    
    return pd.Series(all_embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ### APPLY THE UDF TO COMPUTE CHUNK EMBEDDINGS

# COMMAND ----------

# USE THE UDF
df = spark.table("pdf_raw_chunks").withColumn("embedding", get_embedding(F.col("content")))
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SAVE CHUNK EMBEDDINGS TO DELTA TABLE

# COMMAND ----------

(
    spark.readStream.table("pdf_raw_chunks")
    .withColumn("embedding", get_embedding("content"))
    .selectExpr("path as url", "content", "embedding")
    .writeStream.trigger(availableNow=True)
    .option("checkpointLocation", f"dbfs:{volume_folder}/checkpoints/pdf_chunk")
    .table("sophinea_pdf_documentation")
    .awaitTermination()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## PREVIEW CHUNK EMBEDDINGS

# COMMAND ----------

display(spark.sql("SELECT * FROM sophinea_pdf_documentation WHERE url like '%.pdf' LIMIT 10"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## CREATE SELF-MANAGED VECTOR SEARCH INDEX.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-3.png?raw=true" style="float: center; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC Our dataset is now ready. We chunked the documentation pages into small sections, computed the embeddings and saved it as a Delta Lake table.
# MAGIC
# MAGIC Next, we'll configure Databricks Vector Search to ingest data from this table.
# MAGIC
# MAGIC Vector search index uses a Vector search endpoint to serve the embeddings (you can think about it as your Vector Search API endpoint). <br/>
# MAGIC Multiple Indexes can use the same endpoint. Let's start by creating one.

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## CREATE SELF-MANAGED VECTOR SEARCH USING ENDPOINT

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# THE TABLE WE WOULD LIKE TO INDEX
source_table_fullname = f"{catalog}.{db}.sophinea_pdf_documentation"
# WHERE WE WANT TO STORE OUR INDEX
vs_index_fullname = f"{catalog}.{db}.sophinea_self_managed_vs_index"

if not index_exists(vsc, vs_endpoint_name, vs_index_fullname):
    print(
        f"Creating index {vs_index_fullname} on endpoint {vs_endpoint_name}..."
    )
    vsc.create_delta_sync_index(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_fullname,
        source_table_name=source_table_fullname,
        pipeline_type="TRIGGERED",  # SYNC NEEDS TO BE MANUALLY TRIGGERED 
        primary_key="id",
        embedding_dimension=1024,  # Match your model embedding size (gte)
        embedding_vector_column="embedding",
    )
    # Let's wait for the index to be ready and all our embeddings to be created and indexed
    wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname)
else:
    # Trigger a sync to update our vs content with the new data saved in the table
    wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname)
    vsc.get_index(vs_endpoint_name, vs_index_fullname).sync()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## SEARCHING FOR SIMILAR CONTENT
# MAGIC
# MAGIC That's all we have to do. Databricks will automatically capture and synchronize new entries in your Delta Lake Table.
# MAGIC
# MAGIC Note that depending on your dataset size and model size, index creation can take a few seconds to start and index your embeddings.
# MAGIC
# MAGIC Let's give it a try and search for similar content.
# MAGIC
# MAGIC *Note: `similarity_search` also supports a filters parameter. This is useful to add a security layer to your RAG system: you can filter out some sensitive content based on who is doing the call (for example filter on a specific department based on the user preference).*

# COMMAND ----------

question = "Who's Brian Thamm?"

response = deploy_client.predict(endpoint=embedding_endpoint_name, inputs={"inputs": [question]})
embeddings = response['predictions']  # The response directly contains the embeddings

results = vsc.get_index(vs_endpoint_name, vs_index_fullname).similarity_search(
  query_vector=embeddings[0],  # Use the first (and only) embedding
  columns=["url", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
pprint(docs)

# COMMAND ----------

