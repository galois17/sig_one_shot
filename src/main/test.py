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


MODEL_NAME = "model.onnx"

def check_similarity(session, input_name1, signal1, input_name2, signal2, output_name1, output_name2, threshold=1.0):
    """
    """
    embedding1, embedding2 = session.run([output_name1, output_name2], {input_name1: [signal1], input_name2: [signal2]})
    embedding1 = torch.tensor(embedding1)
    embedding2 = torch.tensor(embedding2)
    distance = F.pairwise_distance(embedding1, embedding2).item()
    print(f"Distance: {distance}")
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

    result = check_similarity(session, input_name1, t1, input_name2, t2, output_name1, output_name2, threshold=0.5)
    if result:
        print("Similar!")
    else:
        print("Not similar!")

if __name__ == '__main__':
    main()