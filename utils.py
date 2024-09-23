from sentence_transformers import SentenceTransformer
import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import requests

load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')
if not pinecone_api_key:
    raise ValueError("Pinecone API key is missing. Please check your .env file.")

pc = Pinecone(
    api_key=pinecone_api_key,
    environment='us-east-1-aws' 
)

if 'aibot' not in pc.list_indexes().names():
    try:
        pc.create_index(
            name='aibot',
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    except Exception as e:
        raise RuntimeError(f"Error creating index: {str(e)}")

index = pc.Index('aibot')

model = SentenceTransformer('all-MiniLM-L6-v2')

hf_api_key = os.getenv("HUGGING_FACE_API_KEY")
if not hf_api_key:
    raise ValueError("Hugging Face API Key is missing. Please check your .env file.")

def find_match(input_text):
    input_em = model.encode(input_text).tolist()
    result = index.query(vector=input_em, top_k=2, include_metadata=True)

    if 'matches' in result and len(result['matches']) >= 2:
        return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']
    return "Not enough matches found."

def query_refiner(conversation_string, query):
    prompt = f"Refine the following query based on the conversation context:\n{conversation_string}\nQuery: {query}"
    
    url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    try:
        response = requests.post(url, headers=headers, json={"inputs": prompt})
        response.raise_for_status()
        refined_query = response.json().get('generated_text', "No generated text found.")
        return refined_query
    except Exception as e:
        raise RuntimeError(f"Error during query refinement: {str(e)}")

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += f"Human: {st.session_state['requests'][i]}\n"
        conversation_string += f"Bot: {st.session_state['responses'][i + 1]}\n"
    return conversation_string
