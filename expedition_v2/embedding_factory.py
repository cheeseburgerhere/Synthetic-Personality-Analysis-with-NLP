import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs', 'embedding.log')),
        logging.StreamHandler()
    ]
)

class EmbeddingGenerator:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logging.info(f"Loading tokenizer and model: {model_name} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.model.eval()
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            raise e

    def get_embedding(self, text_list):
        """
        Generates embeddings for a list of texts.
        Returns numpy array of shape (N, D).
        """
        # Batch processing recommended for large lists, but simplest impl for now
        embeddings = []
        batch_size = 8 # Conservative batch size
        
        logging.info(f"Generating embeddings for {len(text_list)} items...")
        
        with torch.no_grad():
            for i in range(0, len(text_list), batch_size):
                batch_texts = text_list[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # Mean Pooling - standard for getting sentence embeddings from decoder/encoder models often
                # Note: Qwen is a decoder-only LLM usually. 
                # For "embedding" usage, often the last hidden state of the last token (EOS) is used 
                # OR mean pooling of all tokens.
                # We will use Mean Pooling of the last hidden state as a robust default.
                
                last_hidden_state = outputs.last_hidden_state # (B, Seq, Hidden)
                attention_mask = inputs['attention_mask'] # (B, Seq)
                
                # Mask out padding tokens
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                batch_embeddings = sum_embeddings / sum_mask
                
                embeddings.append(batch_embeddings.cpu().numpy())
                
        return np.vstack(embeddings)

def main():
    # Test with a lightweight model if running directly
    # Note: Qwen models might be large to download just for a quick test. 
    # We'll rely on the pipeline runner to call this with real models.
    pass

if __name__ == "__main__":
    main()
