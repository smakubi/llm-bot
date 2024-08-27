# Databricks notebook source
# MAGIC %md
# MAGIC ### INSTALL REQUIRED EXTERNAL LIBRARIES 

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade transformers==4.41.1 pypdf==4.1.0 langchain-text-splitters==0.2.0 databricks-vectorsearch mlflow tiktoken==0.7.0 torch==2.3.0 llama-index==0.10.43
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### LOAD VARIABLES 

# COMMAND ----------

# MAGIC %run ../RESOURCES/02-INIT-ADVANCED $reset_all_data=false 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## EXTRACT PDF CONTENT AS TEXT CHUNKS
# MAGIC
# MAGIC We need to convert the **PDF documents bytes** to **text**, and extract **chunks** from their content.
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-2.png?raw=true" style="float: center" width="500px">
# MAGIC
# MAGIC
# MAGIC
# MAGIC If your PDFs were saved as images, you will need an **OCR to extract** the text.
# MAGIC
# MAGIC Using the **`Unstructured`** library within a **Spark UDF** makes it easy to extract text. 
# MAGIC
# MAGIC
# MAGIC
# MAGIC <br style="clear: both">
# MAGIC
# MAGIC ### SPLITTING BIG DOCUMENTATION PAGE IN SMALLER CHUNKS
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/chunk-window-size.png?raw=true" style="float: center" width="700px">
# MAGIC
# MAGIC In this workshop, some PDFs can be very large, with a lot of text.
# MAGIC
# MAGIC We'll extract the content and then use **llama_index `SentenceSplitter`**, and ensure that each chunk isn't bigger **than 500 tokens**. 
# MAGIC
# MAGIC
# MAGIC The chunk size and chunk overlap depend on the use case and the PDF files. 
# MAGIC
# MAGIC Remember that your **prompt + answer should stay below your model max window size (4096 for llama2)**. 
# MAGIC
# MAGIC
# MAGIC <br/>
# MAGIC <br style="clear: both">
# MAGIC <div style="background-color: #def2ff; padding: 15px;  border-radius: 30px; ">
# MAGIC   <strong>Information</strong><br/>
# MAGIC   Remember that the following steps are specific to your dataset. This is a critical part to building a successful RAG assistant.
# MAGIC   <br/> Always take time to review the chunks created and ensure they make sense and contain relevant information.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's start by extracting text from our PDF.

# COMMAND ----------

# MAGIC %md
# MAGIC ## EXTRACT TEXT FROM PDF

# COMMAND ----------

import warnings
import io
import re
from pypdf import PdfReader

def parse_bytes_pypdf(raw_doc_contents_bytes: bytes):
    try:
        pdf = io.BytesIO(raw_doc_contents_bytes)
        reader = PdfReader(pdf)
        
        def clean_text(text):
            # REPLACE MULTIPLE SPACES OR NEW LINES WITH A SINGLE SPACE 
            text = re.sub(r'\s+', ' ', text)
            # HANDLE HYPHENATED WORDS AT THE END OF LINES 
            text = re.sub(r'-\s+', '', text)
            # REMOVE SPACES BEFORE PUNCTUATION 
            text = re.sub(r'\s+([.,;:!?])', r'\1', text)
            return text.strip()
        
        parsed_content = [clean_text(page.extract_text()) for page in reader.pages]
        return "\n\n".join(parsed_content)  # USE DOUBLE NEWLINE TO SEPARATE PAGES 
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## TEST TEXT EXTRACTION FUNCTION

# COMMAND ----------

with requests.get("https://github.com/smakubi/sophinea-pdf/blob/main/PDFs/OHR_Data_Bootcamp_Solicitation_75N98021Q00038_72021.pdf?raw=true") as pdf:
    doc = parse_bytes_pypdf(pdf.content)
    print(doc)

# COMMAND ----------

# MAGIC %md
# MAGIC This looks great. We'll now wrap it with a text_splitter to avoid having too big pages, and create a Pandas UDF function to easily scale that across multiple nodes.
# MAGIC
# MAGIC *Note that our pdf text isn't clean. To make it nicer, we could use a few extra LLM-based pre-processing steps, asking to remove unrelevant content like the list of chapters and to only keep the core text.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## TEXT SPLITTER PANDAS UDF

# COMMAND ----------

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, set_global_tokenizer
from transformers import AutoTokenizer
from typing import Iterator

# REDUCE ARROW BATCH SIZE AS PDF CAN BE BIG IN MEMOERY 
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

# PANDAS UDF
@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:

    # SET LLAMA2 AS TOKENIZER TO MATCH OUR MODEL SIZE (WILL STAY BELOW GTE 1024 LIMIT) 
    set_global_tokenizer(
        AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    # SENTENCE SPLITTER FROM LLAMA_INDEX TO SPLIT ON SENTENCES 
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=10)

    def extract_and_split(b):
        txt = parse_bytes_pypdf(b)
        if txt is None:
            return []
        nodes = splitter.get_nodes_from_documents([Document(text=txt)])
        return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

# MAGIC %md
# MAGIC ### APPLY THE TEXT SPLITTER UDF

# COMMAND ----------

df = spark.table("pdf_raw").withColumn("content", F.explode(read_as_chunk("content")))
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SAVE TABLE AS CHUNKS

# COMMAND ----------

df.write.mode("append").saveAsTable("pdf_raw_chunks")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## WHAT'S REQUIRED FOR OUR VECTOR SEARCH INDEX
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-vector-search-type.png?raw=true" style="float: center" width="800px">
# MAGIC
# MAGIC Databricks provide multiple types of vector search indexes:
# MAGIC
# MAGIC - **Managed embeddings**: you provide a text column and endpoint name and Databricks synchronizes the index with your Delta table 
# MAGIC - **Self Managed embeddings**: you compute the embeddings and save them as a field of your Delta Table, Databricks will then synchronize the index
# MAGIC - **Direct index**: when you want to use and update the index without having a Delta Table.
# MAGIC
# MAGIC In this demo, we will show you how to setup a **Self-managed Embeddings** index. 
# MAGIC
# MAGIC To do so, we will have to first compute the embeddings of our chunks and save them as a Delta Lake table field as `array&ltfloat&gt`

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Introducing Databricks GTE Embeddings Foundation Model endpoints
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-4.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC Foundation Models are provided by Databricks, and can be used out-of-the-box.
# MAGIC
# MAGIC Databricks supports several endpoint types to compute embeddings or evaluate a model:
# MAGIC - DBRX Instruct, a **foundation model endpoint**, or another model served by databricks (ex: llama2-70B, MPT...)
# MAGIC - An **external endpoint**, acting as a gateway to an external model (ex: Azure OpenAI)
# MAGIC - A **custom**, fined-tuned model hosted on Databricks model service
# MAGIC
# MAGIC Open the [Model Serving Endpoint page](/ml/endpoints) to explore and try the foundation models.
# MAGIC
# MAGIC For this demo, we will use the foundation model `GTE` (embeddings) and `DBRX` (chat). <br/><br/>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-foundation-models.png?raw=true" width="600px" >

# COMMAND ----------

#databricks-gte-large-en"

# COMMAND ----------

# DBTITLE 1,Using Databricks Foundation model GTE as embedding endpoint
from mlflow.deployments import get_deploy_client

# gte-large-en Foundation models are available using the /serving-endpoints/databricks-gtegte-large-en/invocations api.
deploy_client = get_deploy_client("databricks")

## NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(
    endpoint="vs_embedding_model_gte_lg", inputs={"input": ["What is Apache Spark?"]}
)
pprint(embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## FINAL SOPHINEA PDF TABLE

# COMMAND ----------

# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS sophinea_pdf_documentation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY < FLOAT >
# MAGIC )
# MAGIC TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# MAGIC %md
# MAGIC ### Computing the chunk embeddings and saving them to our Delta Table
# MAGIC
# MAGIC The last step is to now compute an embedding for all our documentation chunks. Let's create an udf to compute the embeddings using the foundation model endpoint.
# MAGIC
# MAGIC *Note that this part would typically be setup as a production-grade job, running as soon as a new documentation page is updated. <br/> This could be setup as a Delta Live Table pipeline to incrementally consume updates.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## FUNCTION TO COMPUTE CHUNK EMBEDDINGS

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments

    deploy_client = mlflow.deployments.get_deploy_client("databricks")

    def get_embeddings(batch):
        # Note: this will fail if an exception is thrown during embedding creation (add try/except if needed)
        response = deploy_client.predict(
            endpoint="databricks-gte-large-en", inputs={"input": batch}
        )
        return [e["embedding"] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [
        contents.iloc[i : i + max_batch_size]
        for i in range(0, len(contents), max_batch_size)
    ]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SAVE CHUNK EMBEDDINGS TO DELTA TABLE

# COMMAND ----------

(
    spark.readStream.table("pdf_raw")
    .withColumn("content", F.explode(read_as_chunk("content")))
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

# MAGIC %sql
# MAGIC SELECT * FROM sophinea_pdf_documentation WHERE url like '%.pdf' LIMIT 10

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

# MAGIC %md
# MAGIC ## CREATING VECTOR SEARCH ENDPOINT

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC You can view your endpoint on the [Vector Search Endpoints UI](#/setting/clusters/vector-search). Click on the endpoint name to see all indexes that are served by the endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ## CREATE SELF-MANAGED VECTOR SEARCH USING ENDPOINT

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# The table we'd like to index
source_table_fullname = f"{catalog}.{db}.sophinea_pdf_documentation_experimental"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.sophinea_pdf_experimental_self_managed_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
    print(
        f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}..."
    )
    vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=vs_index_fullname,
        source_table_name=source_table_fullname,
        pipeline_type="TRIGGERED",  # Sync needs to be manually triggered
        primary_key="id",
        embedding_dimension=1024,  # Match your model embedding size (gte)
        embedding_vector_column="embedding",
    )
    # Let's wait for the index to be ready and all our embeddings to be created and indexed
    wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
    # Trigger a sync to update our vs content with the new data saved in the table
    wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
    vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

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

question = """Who's Brian Thamm?"""

response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["url", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
print(docs)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## NEXT STEP: DEPLOY OUR CHATBOT MODEL WITH RAG
# MAGIC
# MAGIC We've seen how Databricks Lakehouse AI makes it easy to ingest and prepare your documents, and deploy a Self Managed Vector Search index on top of it with just a few lines of code and configuration.
# MAGIC
# MAGIC This simplifies and accelerates your data projects so that you can focus on the next step: creating your realtime chatbot endpoint with well-crafted prompt augmentation.
# MAGIC
# MAGIC Open the [02-ADVANCED-CHATBOT-CHAIN]($./02-ADVANCED-CHATBOT-CHAIN) notebook to create and deploy a chatbot endpoint.