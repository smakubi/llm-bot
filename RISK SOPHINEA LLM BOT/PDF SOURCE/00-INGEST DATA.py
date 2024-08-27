# Databricks notebook source
# MAGIC %md
# MAGIC ###  INGEST PDF FROM GITHUB
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC In this example, we will focus on ingesting pdf documents as source for our retrieval process. 
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-0.png?raw=true" style="float: center; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC
# MAGIC For this example, we will add Sophinea previous RFPs to our knowledge database.
# MAGIC
# MAGIC
# MAGIC
# MAGIC Here are all the detailed steps:
# MAGIC
# MAGIC - Use **autoloader** to load the binary PDFs into our first table. 
# MAGIC - Use the **`unstructured`** library  to parse the text content of the PDFs.
# MAGIC - Use **`llama_index`** or **`Langchain`** to split the texts into chuncks.
# MAGIC - **Compute embeddings for the chunks.**
# MAGIC - Save our text chunks + embeddings in a **Delta Lake** table, ready for **Vector Search indexing**.
# MAGIC
# MAGIC
# MAGIC Lakehouse AI not only provides state of the art solutions to accelerate your AI and LLM projects, but also to accelerate data ingestion and preparation at scale, including unstructured data like PDFs.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F03-advanced-app%2F01-PDF-Advanced-Data-Preparation&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F03-advanced-app%2F01-PDF-Advanced-Data-Preparation&version=1">

# COMMAND ----------

# MAGIC %md
# MAGIC ### INSTALL REQUIRED EXTERNAL LIBRARIES 

# COMMAND ----------

# DBTITLE 1,Install required external libraries 
# INSTALL LIBRARIES AND RESTART PYTHON KERNEL
%pip install --quiet --upgrade transformers==4.41.1 pypdf==4.1.0 langchain-text-splitters==0.2.0 databricks-vectorsearch mlflow tiktoken==0.7.0 torch==2.3.0 llama-index==0.10.43
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### LOAD VARIABLES AND HELPER FUNCTIONS

# COMMAND ----------

# MAGIC %run ../RESOURCES/02-INIT-ADVANCED $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## INGEST SOPHINEA RFP PDF AND EXTRACT PAGES
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-1.png?raw=true" style="float: center" width="500px">
# MAGIC
# MAGIC First, let's ingest our PDFs as a Delta Lake table with path urls and content in binary format. 
# MAGIC
# MAGIC We'll use [Databricks Autoloader](https://docs.databricks.com/en/ingestion/auto-loader/index.html) to incrementally ingeset new files, making it easy to incrementally consume billions of files from the data lake in various data formats. Autoloader easily ingests our unstructured PDF data in binary format.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### LOAD PDF INTO VOLUME

# COMMAND ----------

volume_folder = f"/Volumes/{catalog}/{db}/{volume_name}"
upload_pdfs_to_volume(volume_folder+"/sophinea-pdf")
display(dbutils.fs.ls(volume_folder+"/sophinea-pdf"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## INGEST PDF INTO DELTA AS BINARY USING AUTOLOADER

# COMMAND ----------

# READ PDF AS BINARY USING AUTOLOADER
df = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "BINARYFILE")
    .option("pathGlobFilter", "*.pdf")
    .load('dbfs:'+volume_folder+"/sophinea-pdf")
)

# WRITE THE DATA INTO A DELTA TABLE 
(
    df.writeStream.trigger(availableNow=True)
    .option("checkpointLocation", f"dbfs:{volume_folder}/checkpoints/raw_docs")
    .table("pdf_raw")
    .awaitTermination()
)


# COMMAND ----------

# LETS QUERY DATA IN THE DELTA TABLE
display(spark.sql(f"SELECT * FROM pdf_raw LIMIT 10"))