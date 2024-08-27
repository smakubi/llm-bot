# Databricks notebook source
import requests

API_URL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
EMAIL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
USER_NAME = EMAIL.split('@')[0]
URL = f"{API_URL}/api/2.0/clusters/create"


CLUSTER_CONFIG = {
    "cluster_name": f"{USER_NAME}'s LLM Cluster",
    "spark_version": "14.3.x-scala2.12",
    "spark_conf": {
        "spark.databricks.dataLineage.enabled": "true",
        "spark.master": "local[*, 4]"
    },
    "aws_attributes": {
        "first_on_demand": 1,
        "availability": "ON_DEMAND",
        "spot_bid_price_percent": 100,
        "ebs_volume_count": 0
    },
    "node_type_id": "r5d.24xlarge",
    "custom_tags": {
        "project": "LLM",
    },
    "autotermination_minutes": 120,
    "single_user_name": f"{EMAIL}",
    "data_security_mode": "SINGLE_USER",
    "runtime_engine": "STANDARD",
    "autoscale": {
        "min_workers": 2,
        "max_workers": 4
    }
}

response = requests.post(URL, json=CLUSTER_CONFIG, headers={"Authorization": f"Bearer {TOKEN}"})


if response.status_code == 200:
    print("Cluster created successfully")
else:
    print("Failed to create cluster:", response.json())