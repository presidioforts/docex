# Sentence Transformer API

This is a Flask-based API that serves the Sentence Transformer model for generating embeddings and calculating similarities between sentences.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the API

Start the Flask server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### 1. Generate Embeddings
- **Endpoint**: `/encode`
- **Method**: POST
- **Request Body**:
```json
{
    "sentences": ["The weather is lovely today.", "It's so sunny outside!"]
}
```
- **Response**:
```json
{
    "embeddings": [[...], [...]]
}
```

### 2. Calculate Similarities
- **Endpoint**: `/similarity`
- **Method**: POST
- **Request Body**:
```json
{
    "embeddings": [[...], [...]]
}
```
- **Response**:
```json
{
    "similarities": [[...], [...]]
}
```

## Security Notes
- The API is configured to run in production mode (debug=False)
- CORS is enabled for cross-origin requests
- Input validation is implemented for both endpoints
- Error handling is in place for all operations 