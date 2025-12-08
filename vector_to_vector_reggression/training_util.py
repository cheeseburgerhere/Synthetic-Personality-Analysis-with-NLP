import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import pickle
from datetime import datetime

from vector_projection_model import HobbyProjector


# 2. Define the Dataset
class PersonaHobbyDataset(Dataset):
    def __init__(self, persona_vectors, hobby_vectors):
        """
        Args:
            persona_vectors: Numpy array or Tensor of shape (N, 4096)
            hobby_vectors: Numpy array or Tensor of shape (N, 4096)
        """
        self.persona_vectors = torch.FloatTensor(persona_vectors)
        self.hobby_vectors = torch.FloatTensor(hobby_vectors)
        
        assert len(self.persona_vectors) == len(self.hobby_vectors), "Mismatch in number of samples"

    def __len__(self):
        return len(self.persona_vectors)

    def __getitem__(self, idx):
        return self.persona_vectors[idx], self.hobby_vectors[idx]

# 3. Training Loop
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device='cuda'):
    criterion = nn.CosineEmbeddingLoss() # Maximizes similarity
    # Alternatively use MSELoss if magnitude matters: criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model = model.to(device)
    
    best_loss = float('inf')
    
    print(f"Starting training on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for personas, hobbies in train_loader:
            personas = personas.to(device)
            hobbies = hobbies.to(device)
            
            # Target for CosineEmbeddingLoss: 1.0 means we want inputs to be similar
            target = torch.ones(personas.shape[0]).to(device)
            
            optimizer.zero_grad()
            
            outputs = model(personas)
            
            # Note: If outputs are normalized in forward(), make sure hobbies are too if using MSE.
            # For CosineEmbeddingLoss, it computes 1 - cos(x, y).
            loss = criterion(outputs, hobbies, target)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * personas.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for personas, hobbies in val_loader:
                personas = personas.to(device)
                hobbies = hobbies.to(device)
                target = torch.ones(personas.shape[0]).to(device)
                
                outputs = model(personas)
                loss = criterion(outputs, hobbies, target)
                val_loss += loss.item() * personas.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            time = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.state_dict(), f'training_data/best_hobby_projector_{time}.pth')
            print("  Example saved best model.")

    print("Training complete.")
    return model

if __name__ == "__main__":
    # Example Usage:
    # Generate dummy data for demonstration if no real data is loaded
    print("Generating dummy data for testing dimensions (1, 4096)...")
    N_SAMPLES = 100
    DIMENSION = 4096
    
    dummy_personas = np.random.randn(N_SAMPLES, DIMENSION).astype(np.float32)
    dummy_hobbies = np.random.randn(N_SAMPLES, DIMENSION).astype(np.float32)
    
    # Normalize dummy data to simulate embeddings
    dummy_personas /= np.linalg.norm(dummy_personas, axis=1, keepdims=True)
    dummy_hobbies /= np.linalg.norm(dummy_hobbies, axis=1, keepdims=True)
    
    dataset = PersonaHobbyDataset(dummy_personas, dummy_hobbies)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = HobbyProjector(input_dim=DIMENSION, output_dim=DIMENSION)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, val_loader, num_epochs=5, device=str(device))
