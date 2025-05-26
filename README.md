# Sentence Transformer Training Platform
# Open AI Integration with backend for ReactjsUI
A user-friendly platform for training and fine-tuning sentence transformer models using Streamlit UI. This platform allows you to easily train custom sentence transformer models for your specific use case.

## 🌟 Features

- 📊 Interactive Streamlit UI for model training
- 🔄 Fine-tuning capabilities for sentence transformer models
- 📁 CSV-based training data support
- ⚙️ Configurable training parameters
- 💾 Model checkpointing and saving
- 📈 Training progress monitoring

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Technical Documentation](#technical-documentation)
4. [User Guide](#user-guide)
5. [API Documentation](#api-documentation)
6. [Development Guide](#development-guide)
7. [Troubleshooting](#troubleshooting)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/presidioforts/docex.git
cd sentence_transformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

2. Upload your training data CSV file
3. Configure training parameters
4. Start training

## Technical Documentation

### Architecture

The platform consists of three main components:

1. **Model Manager (`model_manager.py`)**
   - Handles model loading and training
   - Manages model checkpointing
   - Provides model evaluation capabilities

2. **Streamlit UI (`streamlit_app.py`)**
   - User interface for training configuration
   - Training progress visualization
   - Model selection and parameter tuning

3. **Training Flow**
   - Data preprocessing
   - Model fine-tuning
   - Checkpoint management

### Model Management

The `ModelManager` class provides the following functionality:

- Model initialization
- Training configuration
- Checkpoint handling
- Model evaluation

### Training Flow

1. Data Loading
2. Parameter Configuration
3. Model Training
4. Checkpoint Creation
5. Model Saving

## User Guide

### Preparing Training Data

Create a CSV file with the following columns:
- `sentence1`: First sentence in the pair
- `sentence2`: Second sentence in the pair
- `score`: Similarity score (0-1)

Example:
```csv
sentence1,sentence2,score
"The cat is on the mat","A cat sits on a mat",0.9
"The weather is nice","It's raining outside",0.1
```

### Using the UI

1. **Upload Data**
   - Click "Upload CSV" button
   - Select your training data file

2. **Configure Training**
   - Set batch size
   - Adjust learning rate
   - Configure epochs
   - Select model checkpoint frequency

3. **Start Training**
   - Click "Start Training" button
   - Monitor progress
   - Wait for completion

### Training Configuration

Recommended settings:
- Batch size: 16-32
- Learning rate: 2e-5
- Epochs: 3-5
- Evaluation steps: 100

## API Documentation

### ModelManager Class

```python
class ModelManager:
    def __init__(self, model_name: str)
    def train(self, train_data: pd.DataFrame, **kwargs)
    def save_model(self, output_path: str)
    def load_model(self, model_path: str)
```

## Development Guide

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings
- Add unit tests

## Troubleshooting

### Common Issues

1. **Training Data Format**
   - Ensure CSV has required columns
   - Check data types
   - Verify score range (0-1)

2. **Memory Issues**
   - Reduce batch size
   - Free unused memory
   - Check available GPU memory

3. **Training Errors**
   - Verify model compatibility
   - Check parameter values
   - Monitor error messages

### Support

For issues and questions:
1. Check existing issues
2. Create a new issue
3. Provide error details

## License

MIT License - See LICENSE file for details

## Supported Models

You can use this platform with various pre-trained sentence transformer models from Hugging Face. Here are some recommended models:

1. **all-MiniLM-L6-v2** (Default)
   - Fast and efficient
   - 384 dimensions
   - Good balance of speed and performance
   - Ideal for: General purpose sentence embeddings

2. **all-mpnet-base-v2**
   - Higher quality embeddings
   - 768 dimensions
   - Better performance but slower
   - Ideal for: When accuracy is critical

3. **paraphrase-multilingual-MiniLM-L12-v2**
   - Multilingual support
   - 384 dimensions
   - Works with 50+ languages
   - Ideal for: Multilingual applications

4. **multi-qa-MiniLM-L6-cos-v1**
   - Optimized for question-answering
   - 384 dimensions
   - Good for semantic search
   - Ideal for: QA and search applications

### Using Different Models

To use a different model:

1. In the Streamlit UI:
   - Select your desired model from the dropdown
   - The platform will automatically download and use the selected model

2. Via API:
```python
from model_manager import ModelManager

# Initialize with your chosen model
model_manager = ModelManager(model_name="all-mpnet-base-v2")

# Train as usual
model_manager.train(train_data=your_data)
```

### Model Selection Guidelines

Choose your model based on your specific needs:

1. **Speed vs Quality**
   - MiniLM models: Faster, smaller, good for production
   - MPNet models: Higher quality, larger, better for accuracy-critical tasks

2. **Language Support**
   - Multilingual models: For non-English or multi-language tasks
   - English-only models: Better performance if you only need English

3. **Resource Constraints**
   - Consider model size and memory requirements
   - Larger models need more GPU memory for training

4. **Domain Specificity**
   - QA models: Better for question-answering tasks
   - Paraphrase models: Better for similarity tasks
   - General models: Good for most use cases 