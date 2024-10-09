import streamlit as st
import boto3
import json
from PyPDF2 import PdfReader
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize SageMaker runtime client
try:
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    endpoint_name = "jumpstart-dft-meta-textgeneration-l-20241009-114629"
    logger.info(f"Successfully initialized SageMaker runtime client for endpoint: {endpoint_name}")
except Exception as e:
    logger.error(f"Failed to initialize SageMaker runtime client: {e}")
    st.error("Failed to initialize SageMaker runtime client. Please check your AWS configuration.")

# Specify the path to your PDF document
pdf_path = "gender_bias.pdf"

# Function to read content from the PDF file
def read_pdf_content(pdf_path):
    content = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    content += text + " "
        logger.info(f"Successfully extracted {len(content)} characters from PDF")
        return content
    except FileNotFoundError:
        error_msg = f"PDF file not found at path: {pdf_path}"
        logger.error(error_msg)
        st.error(error_msg)
        return ""
    except Exception as e:
        error_msg = f"Error reading PDF: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return ""

# Retrieve documents based on user input
def retrieve_documents(query):
    document_content = read_pdf_content(pdf_path)
    if not document_content:
        return []
    return [sent.strip() for sent in document_content.split(".") if sent.strip()]

def get_bot_response(user_input, context):
    if not context.strip():
        return "No valid context extracted from the document."
    
    try:
        prompt = f"Context: {context}\n\nQuestion: {user_input}\n\nAnswer:"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "top_p": 1.0,
                "temperature": 1.0
            }
        }
        logger.info(f"Sending payload to SageMaker: {json.dumps(payload)[:200]}...")

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        response_body = json.loads(response['Body'].read().decode())
        logger.info(f"Received response from SageMaker: {str(response_body)[:200]}...")
        
        # Extract the generated text
        generated_text = response_body.get('generated_text', '')
        
        # Remove quotes from the beginning and end of the generated text
        generated_text = generated_text.strip('"')
        
        # Split the response if it contains a follow-up question
        split_response = re.split(r'\n\nQuestion:', generated_text, 1)
        
        # Take only the first part (the answer to the original question)
        answer = split_response[0].strip()
        
        # Remove the original prompt from the answer
        answer = answer.replace(prompt, '').strip()
        
        if answer:
            return answer
        else:
            return "I'm sorry, I couldn't generate a relevant answer. Please try rephrasing your question."

    except Exception as e:
        error_msg = f"Error invoking SageMaker endpoint: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"

# Streamlit UI
def main():
    st.title("Gender Bias Chatbot")
    st.write("Ask me anything about gender bias!")

    user_input = st.text_input("You:", "")

    if st.button("Send"):
        if user_input:
            with st.spinner("Thinking..."):
                try:
                    retrieved_docs = retrieve_documents(user_input)
                    if not retrieved_docs:
                        st.warning("No content could be extracted from the PDF. Please check the file path and contents.")
                        return

                    context = " ".join(retrieved_docs)
                    MAX_CONTEXT_LENGTH = 2000
                    context = context[:MAX_CONTEXT_LENGTH]
                    
                    bot_response = get_bot_response(user_input, context)
                    st.text_area("Bot:", value=json.dumps(bot_response, indent=2), height=200)
                except Exception as e:
                    logger.error(f"Error in main execution: {e}")
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a message.")

if __name__ == "__main__":
    main()