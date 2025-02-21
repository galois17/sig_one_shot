import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import faiss 
import pandas as pd
import random
import torch.onnx
import onnx
from onnx import helper
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import os
import plotly.graph_objects as go
from plotly.offline import plot  # For offline plotting
from plotly.subplots import make_subplots
import webbrowser

MODEL_NAME = "model.onnx"

class ContrastiveLoss(nn.Module):
    """ 
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        euclidean_distance = nn.functional.pairwise_distance(embedding1, embedding2)
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +  
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)) 
        return loss
    
class SiameseRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(SiameseRNN, self).__init__()
        self.hidden_dim = hidden_dim
        # Keep it simple with Gru
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)  
        self.fc = nn.Linear(hidden_dim, embedding_dim)  # Linear layer 

    def forward_once(self, x):
        output, _ = self.rnn(x) 

        last_timestep = output[:, -1, :] 
        embedding = self.fc(last_timestep)  
        return embedding

    def forward(self, input1, input2):
        embedding1 = self.forward_once(input1)
        embedding2 = self.forward_once(input2)
        return embedding1, embedding2

def save_model(model, input_size, feature_names):
    """ Save model """
     
    dummy_input = torch.randn(input_size, 1).float()
    dummy_input = dummy_input.unsqueeze(0)
    
    
    torch.onnx.export(model, (dummy_input, dummy_input), MODEL_NAME, 
                      input_names=["input1", "input2"], output_names=["output1", "output2"])
    
    
    # Load ONNX model to add metadata
    onnx_model = onnx.load(MODEL_NAME)
    metadata_props = {
        "input_size": str(input_size),
        "feature_order": ",".join(feature_names),
    }

    for key, value in metadata_props.items():
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(onnx_model, MODEL_NAME)

def collate_fn(batch): 
    """ Handle variable sequence lengths """
    input_a = [torch.tensor(x[0], dtype=torch.float32) for x, _ in batch]
    input_b = [torch.tensor(x[1], dtype=torch.float32) for x, _ in batch]
    labels = torch.tensor([y for _, y in batch], dtype=torch.float32)

    input_a = nn.utils.rnn.pad_sequence(input_a, batch_first=True)
    input_b = nn.utils.rnn.pad_sequence(input_b, batch_first=True)

    return input_a, input_b, labels

def main(args):
    f = args.file_path
    if not os.path.exists(f):
        raise ValueError("Parquet file doesn't exist...")
    
    try:
        table = pq.read_table(f)
        df = table.to_pandas()
    except AttributeError:
        all_files = []
        for root, _, files in os.walk(f):
            for file in files:
                if file.endswith(".parquet"):
                    all_files.append(os.path.join(root, file))
        
        df = pd.concat([pq.read_table(f).to_pandas() for f in all_files])

    mappings = {'asco': 0, 'fucus': 1, 'h_asco': 2, 'h_fucus': 3,  'unknown': 4}
    inv_mappings = {v:k for k,v in mappings.items()}

    feature_names = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', \
                'b9', 'b10', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', \
                'd8', 'd9', 'd10', 'ndvi', 'ndwi', 'elev', 
                'energy', 'homogeneity', 'dissimilarity', 'ASM', 'entropy']

    np_data = df[feature_names].to_numpy()
    np_y = df['label'].to_numpy()

    input_dim = 1 
    hidden_dim = 64
    embedding_dim = 32
    learning_rate = 0.001

    if args.epoch:
        num_epochs = args.epoch
    else:
        num_epochs = 100
    batch_size = 32

    model = SiameseRNN(input_dim, hidden_dim, embedding_dim)
    margin = 1.0  # Choose an appropriate margin
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def generate_data(num_samples, seq_len_range):
        """ """
        data = []
        labels = []
        for _ in range(num_samples):
            is_similar = np.random.randint(0, 2)
            label1 = 0
            label2 = 0
            if is_similar == 1:
                label = np.random.randint(0, 3)
                label1 = label2 = label
            else:
                label1 = np.random.randint(0, 3)
                label2 = (label1 + 1) % 3

            idx1= np.random.choice(list(np.where(np_y == label1)[0]), replace=False, size=1)
            idx2 = np.random.choice(list(np.where(np_y == label2)[0]), replace=False, size=1)
            
            input1 = np.reshape(np_data[idx1], (-1,1))
            input2 = np.reshape(np_data[idx2], (-1,1))
            data.append((input1, input2))

            labels.append(is_similar)
        return data, np.array(labels)

    # Variable length...
    train_data, train_labels = generate_data(2000, (10, 24)) # Training data
    val_data, val_labels = generate_data(500, (10, 24)) # Validation data

    train_loader = torch.utils.data.DataLoader(list(zip(train_data, train_labels)), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(list(zip(val_data, val_labels)), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Training loop
    losses = []
    val_auc_scores = []

    def update_plot(epoch, loss, auc, html_file):
        losses.append(loss)
        val_auc_scores.append(auc)
        
        fig = go.Figure()
        
        # Training Loss
        fig.add_trace(go.Scatter(y=losses, mode='lines', name='Training Loss'))
        fig.update_xaxes(title_text='Epoch')
        # fig.update_yaxes(title_text='Loss')
        
        # Validation AUC
        fig.add_trace(go.Scatter(y=val_auc_scores, mode='lines', name='Validation AUC'))
        fig.update_xaxes(title_text='Epoch')
        # fig.update_yaxes(title_text='AUC')
        
        fig.update_layout(
            title_text='Training Progress',
            height=400, width=800,
            template='plotly_dark'
        )

        html_content = f"""
        <html>
        <head>
            <meta http-equiv="refresh" content="2">
        </head>
        <body>
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
        </body>
        </html>
        """
        
        with open(html_file, "w") as f:
            f.write(html_content)
        
    html_file = "training_progress.html"
    for epoch in range(num_epochs):
        epoch_loss = 0
        for input1, input2, label in train_loader:
            optimizer.zero_grad()
            embedding1, embedding2 = model(input1, input2)

            loss = criterion(embedding1, embedding2, label) # Pass the label to the loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch == 0:
            webbrowser.open("file://" + os.path.abspath(html_file))
    
        losses.append(epoch_loss / len(train_loader))

        # Validation loop
        val_losses = []  
        val_distances = [] 
        with torch.no_grad():
            for input1, input2, label in val_loader: 
                embedding1, embedding2 = model(input1, input2)
                distance = nn.functional.pairwise_distance(embedding1, embedding2)  # Calculate distances
                val_distances.extend(distance.tolist()) # Store for AUC

                loss = criterion(embedding1, embedding2, label) 
                val_losses.append(loss.item()) 
        # 1-distances because smaller distance means more similar
        val_auc = roc_auc_score(val_labels, 1-np.array(val_distances))
        val_auc_scores.append(val_auc)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses[-1]:.4f}, Validation AUC: {val_auc:.4f}")

        update_plot(epoch, loss, val_auc, html_file)

    # Switch to eval mode
    model.eval()

    def get_embedding(input_sequence):
        """
        Generates an embedding for a single input sequence.
        """
        with torch.no_grad():  # No gradients needed for inference
            try:
                input_sequence = input_sequence.unsqueeze(0) # Add batch dimension
                embedding = model.forward_once(input_sequence)
                return embedding.squeeze(0) # Remove batch dimension
            except Exception as e: # Handle any potential errors
                print(f"Error getting embedding: {e}")
                return None
            
    # Test with a random signal
    input_dim = 1 # Example
    sequence_length = 15 # Example
    input_sequence = torch.randn(sequence_length, input_dim).float() 

    embedding = get_embedding(input_sequence)
    print(f"An embedding: {embedding}")

    # Save model
    save_model(model, np_data.shape[1], feature_names)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Train one-shot")

    parser.add_argument("-f", "--file_path", type=str, help="path to parquet data")
    parser.add_argument("-e", "--epoch", type=int, help="epoch count")
    parser.add_argument("-x", "--add_texture", action='store_true', default=False, help='add texture')

    args = parser.parse_args()

    main(args)