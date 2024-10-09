from fastapi import FastAPI, UploadFile, File
from typing import List, Dict
from pydantic import BaseModel
import uvicorn

# Import the BiasDetector
from bias_detector import BiasDetector

app = FastAPI()

# Initialize the model
detector = BiasDetector("model_name")  # Replace with actual Llama model

class TextInput(BaseModel):
    text: str

@app.post("/detect_bias/")
def detect_bias(input_data: TextInput):
    """
    Endpoint to detect bias in a given text.
    """
    result = detector.detect_and_suggest_biased_words(input_data.text)
    return result

@app.post("/upload_document/")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a document, detect biases, and return the results.
    """
    content = await file.read()
    text = content.decode('utf-8')
    paragraphs = text.split("\n\n")  # Split into paragraphs or sentences for processing
    
    bias_score = detector.calculate_bias_score(paragraphs)
    results = [detector.detect_and_suggest_biased_words(paragraph) for paragraph in paragraphs]
    
    return {
        "bias_score": bias_score,
        "detection_results": results
    }

@app.get("/health_check/")
def health_check():
    """
    Endpoint to check if the API is running.
    """
    return {"status": "API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
