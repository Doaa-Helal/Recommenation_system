from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from transformers import AutoTokenizer, AutoModel
from pymilvus import (Collection,connections)
from flask import Flask, request, jsonify
from sklearn.pipeline import Pipeline
import torch.nn.functional as F
from dotenv import load_dotenv, dotenv_values
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
import pandas as pd
import warnings
import torch
import os

warnings.filterwarnings("ignore")
#connect to vector database
load_dotenv()
ENDPOINT=os.getenv("milvus_endpoint")
TOKEN=os.getenv("milvus_token")
collection_name=os.getenv("collection_name")
GEN_API_KEY = os.getenv("GOOGLE_API_KEY")


connections.connect(
   uri=ENDPOINT,
   token=TOKEN)
collection = Collection(name=collection_name)

genai.configure(api_key=GEN_API_KEY)


def summariza_image(image):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt_templete="Describe the image in detail. For context,the image is part business model canvas be specific about every thing in the image and in details"

    messages=[
        (
            "user",
            [
                {"type":"text","text":prompt_templete},
                {
                    "type":"image_url",
                    "image_url":{"url":"data:image/jpeg;base64,{image}"}
                },
            ],
        )
    ]

    prompt_msg=ChatPromptTemplate.from_messages(messages)
    chain_img=prompt_msg | model | StrOutputParser()
    img_summ=chain_img.batch(image)
    return img_summ

def lower_transform(text):
    return text.lower()

def remove_excess_whitespace(text):
    stripped_text = text.strip()
    cleaned_text = ' '.join(stripped_text.split())
    return cleaned_text



def search (embeddings,collection):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}    
    results = collection.search(
    
    data=[embeddings], 
    anns_field="vector",  
    output_fields=["email","description"],
    limit=10,
    param=search_params
)
    for hits in results:
        
        emails=[]
        descriptions=[]
        for hit in hits:
            # gets the value of an output field specified in the search request.
            # dynamic fields are supported, but vector fields are not supported yet.    
            emails.append(hit.entity.get('email'))
            descriptions.append(hit.entity.get('description'))

    return emails,descriptions

# Initialize the tokenizer and BERT model


# Load the 'all-roberta-large-v1' model

class BertEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='sentence-transformers/all-roberta-large-v1', device=None):
        # Initialize the device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load the tokenizer and model from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Tokenize the input text
        X=[X]
        encoded_input = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        # Generate token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform mean pooling to get sentence embeddings
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.cpu().numpy().flatten()

# Initialize the transformers
lowercase_transformer = FunctionTransformer(lower_transform, validate=False)
whitespace_transformer = FunctionTransformer(remove_excess_whitespace, validate=False)
bert_embedding_transformer = BertEmbeddingTransformer()

# Create the pipeline
embeddings_pipeline = Pipeline([
    ('lowercase', lowercase_transformer),
    ('whitespace', whitespace_transformer),
    ('bert_embedding', bert_embedding_transformer)
])


#defining endpoints
app = Flask(__name__)

@app.route('/store', methods=['POST'])
def store_data():
    try:
        # Get the JSON data from the request
        text = request.get_json()  # Ensures that the JSON is parsed or raises an error
        
        # Validate if the input is a dictionary
        if not isinstance(text, dict):
            return jsonify({"error": "Input data must be a JSON object"}), 400
        
        # Extract fields from the JSON
        title = text.get("title", "").strip()
        id=text.get("id","").strip()
        description = text.get("description", "").strip()
        category = text.get("category", "").strip()
        resources_required = text.get("resourcesRequired", [])
        
        if not description or not category:
            return jsonify({"error": "Missing required fields: 'description' or 'category'"}), 400
        
        if isinstance(resources_required, list):
            for resource in resources_required:
                resource_type = resource.get("type", "Unknown Type").strip()
                resource_details = resource.get("details", [])
                
                # Ensure details is a list
                if not isinstance(resource_details, list):
                    resource_details = [str(resource_details)]
                
                # Format and append resource details to the description
                resource_details_str = ", ".join(map(str, resource_details))
                description += f"\n- {resource_type}: {resource_details_str}"
        else:
            return jsonify({"error": "Invalid format for 'resourcesRequired', must be a list"}), 400
        
        # Process embeddings
        embeddings = embeddings_pipeline.transform(description)
        embeddings_list = embeddings.tolist()
        # Prepare data for insertion
        data_row = {
            "vector": embeddings_list,
            "email": title,
            "description": description,
          #  "category": category
        }
        
        #Insert into collection
        collection.insert([data_row])
        collection.flush()
        
        return jsonify({"status": "stored", "data": data_row}), 200

    except Exception as x:
        return jsonify({"error": str(x)}), 500



@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    try:
        text = request.get_json()  # Ensures that the JSON is parsed or raises an error            
       # Validate if the input is a dictionary
        if not isinstance(text, dict):
            return jsonify({"error": "Input data must be a JSON object"}), 400
        
        # Extract fields from the JSON
        description = text.get("description", "").strip()
        category = text.get("category", "").strip()
        resources_required = text.get("resourcesRequired", [])
        
        if not description or not category:
            return jsonify({"error": "Missing required fields: 'description' or 'category'"}), 400
        
        if isinstance(resources_required, list):
            for resource in resources_required:
                resource_type = resource.get("type", "Unknown Type").strip()
                resource_details = resource.get("details", [])
                
                # Ensure details is a list
                if not isinstance(resource_details, list):
                    resource_details = [str(resource_details)]
                
                # Format and append resource details to the description
                resource_details_str = ", ".join(map(str, resource_details))
                description += f"\n- {resource_type}: {resource_details_str}"
        else:
            return jsonify({"error": "Invalid format for 'resourcesRequired', must be a list"}), 400
        
        embeddings = embeddings_pipeline.transform(description)  
        emails=[]
        descriptions=[]
        emails,descriptions=search(embeddings,collection)
        return jsonify({"emails":emails,"descriptions":descriptions })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/update', methods=['POST'])
def update_data():
    text = request.get_json()
    data = pd.DataFrame([text])
    
    if "email" not in data.columns or ("description" not in data.columns and "image" not in data.columns):
        return jsonify({"error": "Input JSON must contain 'email' and 'description' or image fields"}), 400
    
    email = data["email"].values[0]
    descriptions = data["description"].values[0]
    image=data["image"].values[0]
    img_sum=img_sum(image)
    descriptions+="and the business model canvas is"+img_sum
    embeddings = embeddings_pipeline.transform(descriptions) 
    try:
        data_rows = []
        data_rows.extend([
        {"vector": embeddings,
            "email":email,
            "description":descriptions}
        ])
        collection.upsert(data_rows)
        collection.flush()


        return jsonify({"status":"updated"})
    except Exception as x:
        return jsonify({"error": str(x)}),500    


@app.route('/delete', methods=['POST'])
def delete_data():
    try:
        text = request.get_json()
        data = pd.DataFrame([text])
        
        if "email" not in data.columns :
            return jsonify({"error": "Input JSON must contain 'email'  field"}), 400
        
        email = data["email"].values[0]
        expre=f'email == "{email}"' 
        results = collection.query(expre, output_fields=["email"], limit=1)
        if len(results)==0:
            return jsonify({"error": "No data found for the given email."}), 500
        
        res = collection.delete(
        
            expr=expre 
                    
        )
        return jsonify({"status":"deleted"})
 
    except Exception as e:
        return jsonify({"error": str(e)}), 501

    
if __name__ == "__main__":
    app.run(debug=True)
