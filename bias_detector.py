from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Union, Tuple, List
from PIL import Image
import pytesseract
from moviepy.editor import VideoFileClip
import cv2

class MultilingualBiasDetector:
    """
    A class to detect bias in text, images, and videos across multiple languages.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B", device: str = "cuda"):
        """
        Initialize the multilingual bias detector.
        
        Args:
            model_name (str): Name of the pretrained model to use
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)
        
        # Initialize translation pipelines
        self.translation_pipelines = {
            'ar': pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar"),
            'sw': pipeline("translation", model="Helsinki-NLP/opus-mt-en-swh"),
        }
    
    def translate_text(self, text: str, target_lang: str) -> str:
        """
        Translate text to the target language.
        
        Args:
            text (str): Input text
            target_lang (str): Target language code ('ar' for Arabic, 'sw' for Swahili)
        
        Returns:
            Translated text
        """
        if target_lang not in self.translation_pipelines:
            raise ValueError(f"Unsupported target language: {target_lang}")
        
        return self.translation_pipelines[target_lang](text)[0]['translation_text']
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            Extracted text
        """
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    
    def extract_text_from_video(self, video_path: str) -> List[str]:
        """
        Extract text from video frames.
        
        Args:
            video_path (str): Path to the video file
        
        Returns:
            List of extracted text from frames
        """
        texts = []
        video = VideoFileClip(video_path)
        
        for frame in video.iter_frames(fps=1):  # Process 1 frame per second
            frame_pil = Image.fromarray(frame)
            text = pytesseract.image_to_string(frame_pil)
            if text.strip():
                texts.append(text)
        
        return texts

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
        
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """
        Train the model using the provided datasets.
        """
        training_args = TrainingArguments(
            output_dir='./results',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            evaluation_strategy="epoch",
            logging_dir='./logs',
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        trainer.train()
    
    def predict(self, text: str, language: str = 'en') -> Dict[str, Union[str, float]]:
        """
        Make predictions on whether the text is biased or unbiased.
        
        Args:
            text (str): Input text for bias detection
            language (str): Language of the input text ('en', 'ar', or 'sw')
        
        Returns:
            Dictionary containing the prediction and confidence score
        """
        if language != 'en':
            # Translate to English for processing
            text = self.translate_text(text, 'en')
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        prediction = torch.argmax(logits, dim=1)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        result = {
            "prediction": "biased" if prediction.item() == 1 else "unbiased",
            "confidence": probabilities[0][prediction.item()].item()
        }
        
        if language != 'en':
            # Translate result back to original language
            result["prediction_translated"] = self.translate_text(result["prediction"], language)
        
        return result

def main():
    """
    Main function to demonstrate the usage of the multilingual bias detector.
    """
    try:
        detector = MultilingualBiasDetector()
        
        # Example with text in different languages
        texts = {
            'en': "All individuals should have equal opportunities.",
            'ar': "يجب أن يحصل الجميع على فرص متساوية.",
            'sw': "Watu wote wanapaswa kuwa na fursa sawa."
        }
        
        for lang, text in texts.items():
            result = detector.predict(text, language=lang)
            print(f"Language: {lang}")
            print(f"Text: {text}")
            print(f"Prediction: {result['prediction']}")
            if 'prediction_translated' in result:
                print(f"Prediction (translated): {result['prediction_translated']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print()
        
        # Example with image
        image_text = detector.extract_text_from_image("path/to/your/image.jpg")
        image_result = detector.predict(image_text)
        print(f"Image text prediction: {image_result['prediction']}")
        
        # Example with video
        video_texts = detector.extract_text_from_video("path/to/your/video.mp4")
        for i, text in enumerate(video_texts):
            video_result = detector.predict(text)
            print(f"Video frame {i} prediction: {video_result['prediction']}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()