from pymilvus import (
   FieldSchema, DataType, 
   CollectionSchema, Collection,connections)
from dotenv import load_dotenv, dotenv_values
import os

load_dotenv()
ENDPOINT=os.getenv("milvus_endpoint")
TOKEN=os.getenv("milvus_token")
connections.connect(
   uri=ENDPOINT,
   token=TOKEN)
## 1. Define a minimum expandable schema.
fields = [
   FieldSchema("vector", DataType.FLOAT_VECTOR, dim=1024),
   FieldSchema("email", DataType.VARCHAR,max_length=100, is_primary=True),
   FieldSchema("description", DataType.VARCHAR,max_length=10000),
]

schema = CollectionSchema(
   fields,
   enable_dynamic_field=True,
)

## 2. Create a collection.
mc = Collection("GDG2", schema)

## 3. Index the collection.
mc.create_index(
   field_name="vector",
   index_params={
       "Index_type": "AUTOINDEX",
       "Metric_type": "L2",
       }
)

mc.load()