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
import requests

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
