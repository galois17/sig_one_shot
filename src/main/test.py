import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
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
import onnxruntime as ort
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

MODEL_NAME = "model.onnx"
FAISS_INDEX_FILE = "faiss_index.idx"

def check_similarity(session, index, input_name1, signal1, input_name2, signal2, output_name1, output_name2, threshold=1.0):
    """
    """
    embedding1, embedding2 = session.run([output_name1, output_name2], {input_name1: [signal1], input_name2: [signal2]})
    embedding1 = torch.tensor(embedding1)
    embedding2 = torch.tensor(embedding2)
    distance = F.pairwise_distance(embedding1, embedding2).item()

    embedding_np1 = embedding1.numpy().reshape(1, -1)
    embedding_np2 = embedding2.numpy().reshape(1, -1)

    # Search for the nearest neighbor (k=1)
    D1, I1 = index.search(
        embedding_np1, k=1
    )  

    D2, I2 = index.search(
        embedding_np2, k=1
    ) 

    print(f"Distance: {distance}, embedding1 is close to index: {I1[0][0]}, embedding2 is close to index: {I2[0][0]}")
    print("  Look up the indices in the raw data used to build the vector db to get labels")
    return distance <= threshold  

def main():
    onnx_model = onnx.load(MODEL_NAME)
    session = ort.InferenceSession(MODEL_NAME)

    metadata = {meta.key: meta.value for meta in onnx_model.metadata_props}

    print("Model Metadata:")
    print(f"Input Size: {metadata.get('input_size', 'N/A')}")
    print(f"Feature Order: {metadata.get('feature_order', 'N/A')}")

    t1 = torch.randn(28, 1).float() 
    t2 = torch.randn(28, 1).float()

    # Run inference
    input_name1 = session.get_inputs()[0].name
    input_name2 = session.get_inputs()[1].name
    print(f"Input name: {input_name1}, {input_name2}")
    output_name1 = session.get_outputs()[0].name
    output_name2 = session.get_outputs()[1].name

    loaded_index = faiss.read_index(FAISS_INDEX_FILE)

    result = check_similarity(session, loaded_index, input_name1, t1, input_name2, t2, output_name1, output_name2, threshold=0.5)
    if result:
        print("The two signals are similar!")
    else:
        print("The two signals are NOT similar!")

if __name__ == '__main__':
    main()