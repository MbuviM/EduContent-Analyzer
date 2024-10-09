# Bias Detector using Llama Language Model
from transformers import AutoTokenizer, LlamaForSequenceClassification
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Union, Tuple

class LlamaBiasDetector:
    """
    A class to detect gender bias in text using a fine-tuned language model.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda"):
        """
        Initialize the bias detector with a specified model.
        
        Args:
            model_name (str): Name of the pretrained model to use
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = LlamaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,  # Binary classification: biased or unbiased
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Resize token embeddings for classification
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
    def prepare_dataset(self, data_path: Union[str, pd.DataFrame]) -> Tuple[Dataset, Dataset]:
        """
        Prepare dataset from a CSV file or pandas DataFrame.
        
        Args:
            data_path: Path to CSV file or pandas DataFrame with 'text' and 'label' columns
        
        Returns:
            Tuple of train and validation datasets
        """
        if isinstance(data_path, str):
            df = pd.read_csv(data_path)
        elif isinstance(data_path, pd.DataFrame):
            df = data_path
        else:
            raise ValueError("Input must be a path to a CSV file or a pandas DataFrame.")
        
        # Splitting the data into train and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Create datasets from pandas DataFrame
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """
        Train the model using the provided datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        training_args = TrainingArguments(
            output_dir='./results',
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            evaluation_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        trainer.train()
    
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Make predictions on whether the text is biased or unbiased.
        
        Args:
            text (str): Input text for bias detection
        
        Returns:
            Dictionary containing the prediction ('biased' or 'unbiased') and the confidence score
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        prediction = torch.argmax(logits, dim=1)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        return {
            "prediction": "biased" if prediction.item() == 1 else "unbiased",
            "confidence": probabilities[0][prediction.item()].item()
        }

def main():
    """
    Main function to demonstrate the usage of the bias detector.
    """
    try:
        # Load Wino Bias dataset
        ds_anti = load_dataset("uclanlp/wino_bias", "type2_anti")
        ds_pro = load_dataset("uclanlp/wino_bias", "type2_pro")
        
        # Initialize detector
        detector = LlamaBiasDetector()
        
        # Use a subset of the anti-bias dataset for demonstration
        example_data = pd.DataFrame({
            "text": ds_anti['train']['sentence'][:6],
            "label": [1 if "female" in text else 0 for text in ds_anti['train']['sentence'][:6]]  # Simplified labeling
        })
        
        train_dataset, val_dataset = detector.prepare_dataset(example_data)
        
        # Train the model
        detector.train(train_dataset, val_dataset)
        
        # Example predictions
        test_texts = [
            "Women are not suited for leadership positions.",
            "All individuals should have equal opportunities in education."
        ]
        
        for text in test_texts:
            result = detector.predict(text)
            print(f"Text: {text}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print()
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
