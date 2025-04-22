import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
from model_manager import model_manager
import time
import os

# Set page config
st.set_page_config(
    page_title="Sentence Transformer Training",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = model_manager.get_current_model()
    st.session_state.model_name = model_manager.get_current_model_info()['name']

if 'training_data' not in st.session_state:
    st.session_state.training_data = None

if 'training_status' not in st.session_state:
    st.session_state.training_status = "Not Started"

# Sidebar for model selection
st.sidebar.title("Model Configuration")
model_name = st.sidebar.selectbox(
    "Select Model",
    options=list(model_manager.get_available_models().keys()),
    index=0
)

if model_name != st.session_state.model_name:
    model_manager.switch_model(model_name)
    st.session_state.model = model_manager.get_current_model()
    st.session_state.model_name = model_name
    st.sidebar.success(f"Switched to {model_name}")

# Display model info
model_info = model_manager.get_current_model_info()
st.sidebar.markdown("### Model Information")
st.sidebar.json(model_info)

# Main content
st.title("Sentence Transformer Training")

# File upload section
st.header("1. Upload Training Data")
uploaded_file = st.file_uploader("Upload your training data (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.training_data = df
        st.success("Data uploaded successfully!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

# Training configuration
if st.session_state.training_data is not None:
    st.header("2. Configure Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=16)
        epochs = st.number_input("Number of Epochs", min_value=1, max_value=100, value=3)
        learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=2e-5, format="%e")
    
    with col2:
        warmup_steps = st.number_input("Warmup Steps", min_value=0, max_value=1000, value=100)
        evaluation_steps = st.number_input("Evaluation Steps", min_value=100, max_value=5000, value=1000)

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Training button
    if st.button("Start Training"):
        try:
            st.session_state.training_status = "Training in Progress"
            
            # Prepare training data
            train_examples = []
            for _, row in st.session_state.training_data.iterrows():
                train_examples.append(InputExample(
                    texts=[row['text1'], row['text2']],
                    label=float(row['similarity'])
                ))
            
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
            train_loss = losses.CosineSimilarityLoss(model=st.session_state.model)
            
            # Training progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training loop
            for epoch in range(epochs):
                status_text.text(f"Epoch {epoch + 1}/{epochs}")
                st.session_state.model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=1,
                    warmup_steps=warmup_steps,
                    evaluator=None,
                    evaluation_steps=evaluation_steps,
                    output_path=f"output/checkpoint-{epoch+1}",
                    show_progress_bar=True
                )
                progress_bar.progress((epoch + 1) / epochs)
            
            st.session_state.training_status = "Training Completed"
            st.success("Training completed successfully!")
            
        except Exception as e:
            st.session_state.training_status = "Training Failed"
            st.error(f"Training failed: {str(e)}")

# Display training status
st.header("3. Training Status")
st.info(f"Current Status: {st.session_state.training_status}")

# Model evaluation section
if st.session_state.training_status == "Training Completed":
    st.header("4. Model Evaluation")
    
    # Add evaluation metrics and visualization here
    st.write("Evaluation metrics will be displayed here")
    
    # Option to save the model
    if st.button("Save Model"):
        try:
            save_path = "output/final_model"
            os.makedirs(save_path, exist_ok=True)
            st.session_state.model.save(save_path)
            st.success(f"Model saved to {save_path}")
        except Exception as e:
            st.error(f"Error saving model: {str(e)}") 