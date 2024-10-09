# Chatbot for educating masses on gender inequality, gender bias and sensitivity

from fastapi import FastAPI
import boto3
import json

app = FastAPI()

# Initialize SageMaker runtime client
sagemaker_client = boto3.client('sagemaker-runtime')

model_endpoint = "jumpstart-dft-meta-textgeneration-l-20241009-114629"  # Update with your endpoint

@app.post("/chat")
async def chat_with_llama(request: dict):
    user_input = request.get("text")

    # Prepare the payload to send to the SageMaker endpoint
    payload = json.dumps({"inputs": user_input})

    try:
        # Invoke the SageMaker endpoint
        response = sagemaker_client.invoke_endpoint(
            EndpointName=model_endpoint,
            ContentType="application/json",
            Body=payload
        )

        # Parse the response
        result = json.loads(response["Body"].read().decode())
        bot_response = result['generated_text']  # Adjust according to your model's output

    except Exception as e:
        return {"error": str(e)}

    return {"response": bot_response}
