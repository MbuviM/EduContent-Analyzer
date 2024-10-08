from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from transformers import (
    LlamaTokenizer, 
    LlamaForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
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
        self.tokenizer = AutoModelForCausalLM.from_pretrained(model_name)
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
            raise ValueError("data_path must be a string path to CSV or a pandas DataFrame")
        
        # Ensure required columns exist
        required_columns = ['text', 'label']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        return train_dataset, val_dataset
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenize the input texts.
        
        Args:
            examples: Dictionary containing texts to tokenize
        
        Returns:
            Dictionary of tokenized inputs
        """
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    def compute_metrics(self, eval_pred: Tuple) -> Dict:
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Tuple of predictions and labels
        
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = np.mean(predictions == labels)
        
        return {"accuracy": accuracy}
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
              output_dir: str = "./llama_bias_detector") -> None:
        """
        Train the bias detection model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save the trained model
        """
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_val = val_dataset.map(self.tokenize_function, batched=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_steps=100
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        
        # Save the model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def predict(self, text: str) -> Dict:
        """
        Predict whether a given text contains gender bias.
        
        Args:
            text: Input text to analyze
        
        Returns:
            Dictionary containing prediction and confidence score
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        
        return {
            "prediction": "biased" if prediction.item() == 1 else "unbiased",
            "confidence": probabilities[0][prediction.item()].item()
        }

def prepare_example_data() -> pd.DataFrame:
    """
    Create a small example dataset for demonstration.
    
    Returns:
        pandas DataFrame containing example texts and labels
    """
    texts = [
        "All scientists should strive for excellence in their work.",
        "Female scientists often struggle with complex equations.",
        "Teachers of all genders can inspire students.",
        "Male nurses are unusual in the medical field.",
        "Students should choose their careers based on their interests.",
        "Girls are naturally better at languages than mathematics."
    ]
    
    labels = [0, 1, 0, 1, 0, 1]  # 0 for unbiased, 1 for biased
    
    return pd.DataFrame({
        "text": texts,
        "label": labels
    })

def main():
    """
    Main function to demonstrate the usage of the bias detector.
    """
    try:
        # Initialize detector
        detector = LlamaBiasDetector()
        
        # Prepare example dataset
        example_data = prepare_example_data()
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