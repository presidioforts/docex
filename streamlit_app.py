"""
Streamlit application for training Sentence Transformer models.
This app provides a user-friendly interface for:
1. Uploading training data
2. Configuring model parameters
3. Training the model
4. Evaluating and saving the results

Data Requirements:
- CSV file with columns: 'text1', 'text2', 'similarity'
- text1, text2: Pairs of sentences to compare
- similarity: Float value between 0 and 1 indicating sentence similarity
  - 1.0: Sentences are semantically identical
  - 0.0: Sentences are completely different

Example CSV format:
text1,text2,similarity
"The cat is sleeping","A cat naps on the bed",0.9
"I love pizza","The weather is nice",0.1

Training Process:
1. Model loads pre-trained weights
2. Fine-tunes on your specific sentence pairs
3. Saves checkpoints during training
4. Produces final model optimized for your domain
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
from model_manager import model_manager
import time
import os

# Set page configuration for the Streamlit app
# This configures the basic appearance and layout of the web interface
st.set_page_config(
    page_title="Sentence Transformer Training",
    page_icon="🤖",
    layout="wide"  # Use wide layout for better visibility of components
)

# Initialize session state variables
# Session state persists data between reruns of the Streamlit app
if 'model' not in st.session_state:
    # Load the default model from model_manager
    st.session_state.model = model_manager.get_current_model()
    st.session_state.model_name = model_manager.get_current_model_info()['name']

if 'training_data' not in st.session_state:
    # Initialize training data as None until user uploads a file
    st.session_state.training_data = None

if 'training_status' not in st.session_state:
    # Track the current status of model training
    st.session_state.training_status = "Not Started"

# Sidebar for model selection and configuration
st.sidebar.title("Model Configuration")

# Dropdown to select different sentence transformer models
model_name = st.sidebar.selectbox(
    "Select Model",
    options=list(model_manager.get_available_models().keys()),
    index=0,  # Default to first model in the list
    help="Choose a pre-trained model to fine-tune. Different models have different trade-offs between speed and accuracy."
)

# Handle model switching when user selects a different model
if model_name != st.session_state.model_name:
    # Switch to the newly selected model
    model_manager.switch_model(model_name)
    st.session_state.model = model_manager.get_current_model()
    st.session_state.model_name = model_name
    st.sidebar.success(f"Switched to {model_name}")

# Display current model information in the sidebar
model_info = model_manager.get_current_model_info()
st.sidebar.markdown("### Model Information")
st.sidebar.json(model_info)

# Main content area
st.title("Sentence Transformer Training")

# Section 1: Data Upload
st.header("1. Upload Training Data")
st.markdown("""
### Data Format Requirements:
- CSV file with three columns: `text1`, `text2`, `similarity`
- `text1` and `text2`: Pairs of sentences to compare
- `similarity`: Score between 0 and 1
  - 1.0 means sentences are semantically identical
  - 0.0 means completely different meanings
""")

uploaded_file = st.file_uploader(
    "Upload your training data (CSV)", 
    type=['csv'],  # Only allow CSV files
    help="Upload a CSV file containing sentence pairs and their similarity scores"
)

# Handle file upload and data validation
if uploaded_file is not None:
    try:
        # Read and display the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_columns = ['text1', 'text2', 'similarity']
        if not all(col in df.columns for col in required_columns):
            st.error("CSV must contain columns: 'text1', 'text2', 'similarity'")
        # Validate similarity scores
        elif not df['similarity'].between(0, 1).all():
            st.error("Similarity scores must be between 0 and 1")
        else:
            st.session_state.training_data = df
            st.success("Data uploaded successfully!")
            st.markdown("### Preview of Training Data:")
            st.dataframe(df.head())  # Show first few rows of the data
            
            # Display dataset statistics
            st.markdown("### Dataset Statistics:")
            st.write(f"- Number of sentence pairs: {len(df)}")
            st.write(f"- Average similarity score: {df['similarity'].mean():.2f}")
            st.write(f"- Similarity score distribution:")
            hist_data = np.histogram(df['similarity'], bins=10, range=(0,1))
            st.bar_chart(pd.DataFrame({
                'Similarity Score': hist_data[1][:-1],
                'Count': hist_data[0]
            }).set_index('Similarity Score'))
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

# Section 2: Training Configuration
# Only show if training data has been uploaded
if st.session_state.training_data is not None:
    st.header("2. Configure Training")
    
    st.markdown("""
    ### Parameter Guidelines:
    - **Batch Size**: Larger values use more memory but train faster
    - **Epochs**: More epochs may improve results but take longer
    - **Learning Rate**: Lower values are more stable but train slower
    - **Warmup Steps**: Helps stabilize early training
    - **Evaluation Steps**: How often to evaluate model during training
    """)
    
    # Split configuration into two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic training parameters
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=128,
            value=16,
            help="Number of samples processed before model is updated. Larger values use more memory but train faster."
        )
        epochs = st.number_input(
            "Number of Epochs",
            min_value=1,
            max_value=100,
            value=3,
            help="Number of complete passes through the training dataset. More epochs may improve results but take longer."
        )
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-2,
            value=2e-5,
            format="%e",
            help="Step size for model parameter updates. Lower values are more stable but train slower."
        )
    
    with col2:
        # Advanced training parameters
        warmup_steps = st.number_input(
            "Warmup Steps",
            min_value=0,
            max_value=1000,
            value=100,
            help="Number of steps for learning rate warmup. Helps stabilize early training."
        )
        evaluation_steps = st.number_input(
            "Evaluation Steps",
            min_value=100,
            max_value=5000,
            value=1000,
            help="How often to evaluate model during training. Lower values give more frequent updates but slow training."
        )

    # Create directory for saving model checkpoints and final model
    os.makedirs("output", exist_ok=True)

    # Training initiation button
    if st.button("Start Training"):
        try:
            st.session_state.training_status = "Training in Progress"
            
            # Convert training data to SentenceTransformer format
            train_examples = []
            for _, row in st.session_state.training_data.iterrows():
                train_examples.append(InputExample(
                    texts=[row['text1'], row['text2']],  # Pair of sentences
                    label=float(row['similarity'])       # Similarity score
                ))
            
            # Create data loader for batch processing
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,  # Shuffle data for better training
                batch_size=batch_size
            )
            
            # Initialize loss function for training
            # CosineSimilarityLoss is used for sentence similarity tasks
            train_loss = losses.CosineSimilarityLoss(model=st.session_state.model)
            
            # Setup progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training loop
            for epoch in range(epochs):
                status_text.text(f"Epoch {epoch + 1}/{epochs}")
                
                # Train for one epoch
                st.session_state.model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=1,
                    warmup_steps=warmup_steps,
                    evaluator=None,
                    evaluation_steps=evaluation_steps,
                    output_path=f"output/checkpoint-{epoch+1}",
                    show_progress_bar=True
                )
                
                # Update progress bar
                progress_bar.progress((epoch + 1) / epochs)
            
            # Update training status
            st.session_state.training_status = "Training Completed"
            st.success("Training completed successfully!")
            
        except Exception as e:
            # Handle training errors
            st.session_state.training_status = "Training Failed"
            st.error(f"Training failed: {str(e)}")

# Section 3: Training Status Display
st.header("3. Training Status")
st.info(f"Current Status: {st.session_state.training_status}")

# Section 4: Model Evaluation and Saving
# Only show if training has completed
if st.session_state.training_status == "Training Completed":
    st.header("4. Model Evaluation")
    
    st.markdown("""
    ### Model Usage After Training:
    1. Click 'Save Model' to save the trained model
    2. The model will be saved in the 'output/final_model' directory
    3. You can load this model later using:
    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('output/final_model')
    ```
    """)
    
    # Model saving functionality
    if st.button("Save Model"):
        try:
            # Save the trained model to disk
            save_path = "output/final_model"
            os.makedirs(save_path, exist_ok=True)
            st.session_state.model.save(save_path)
            st.success(f"Model saved to {save_path}")
            
            # Display model usage example
            st.markdown("""
            ### Example Usage:
            ```python
            # Load your saved model
            model = SentenceTransformer('output/final_model')
            
            # Get embeddings for new sentences
            sentences = ['Your sentence here']
            embeddings = model.encode(sentences)
            ```
            """)
        except Exception as e:
            st.error(f"Error saving model: {str(e)}") 