import os
import requests
import streamlit as st
from streamlit_chat import message
from utils import find_match, query_refiner, get_conversation_string

st.title("Ssk-aibot")

hf_api_key = os.getenv("HUGGING_FACE_API_KEY")
if not hf_api_key:
    st.error("Hugging Face API Key not found. Please set the environment variable.")
    st.stop()

st.success("Hugging Face API Key loaded successfully.")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hello! I am Ssk. How can I help you?"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

query = st.text_input("Query:", key="input")
if query:
    with st.spinner("Wait, answering..."):
        conversation_string = get_conversation_string()
        refined_query = query_refiner(conversation_string, query)

        st.subheader("Refined Query:")
        st.write(refined_query)

        context = find_match(refined_query)

        url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
        headers = {"Authorization": f"Bearer {hf_api_key}"}

        try:
            response = requests.post(url, headers=headers, json={"inputs": f"Context:\n{context}\n\nQuery:\n{query}"})
            response.raise_for_status()
            response_content = response.json().get('generated_text', "No response text found.")  # Adjust based on response format
        except Exception as e:
            response_content = "Sorry, there was an error processing your request."
            st.error(f"Error: {str(e)}")

    st.session_state['requests'].append(query)
    st.session_state['responses'].append(response_content)

if st.session_state['responses']:
    for i in range(len(st.session_state['responses'])):
        message(st.session_state['responses'][i], key=str(i))
        if i < len(st.session_state['requests']):
            message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')
