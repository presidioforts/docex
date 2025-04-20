from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import time

app = Flask(__name__)
CORS(app)

# Initialize the model - using a model that doesn't require safetensors
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Mock DevOps error database (in production, this would come from Splunk)
devops_errors = {
    "npm": [
        {
            "error_log": "npm ERR! code ERESOLVE\nnpm ERR! ERESOLVE unable to resolve dependency tree",
            "category": "dependency",
            "resolution": "1. Check package.json versions\n2. Run npm install --legacy-peer-deps\n3. Clear npm cache",
            "severity": "high",
            "frequency": 0.8
        },
        # Add more npm errors
    ],
    "gradle": [
        {
            "error_log": "FAILURE: Build failed with an exception.\n* What went wrong:\nExecution failed for task ':app:compileDebugJavaWithJavac'.",
            "category": "compilation",
            "resolution": "1. Check Java version\n2. Increase heap space\n3. Clean and rebuild",
            "severity": "medium",
            "frequency": 0.6
        },
        # Add more gradle errors
    ],
    "docker": [
        {
            "error_log": "ERROR: failed to solve: failed to compute cache key: failed to calculate checksum",
            "category": "build",
            "resolution": "1. Check Dockerfile syntax\n2. Verify network connectivity\n3. Clear Docker cache",
            "severity": "high",
            "frequency": 0.7
        },
        # Add more docker errors
    ],
    "kubernetes": [
        {
            "error_log": "Error from server (NotFound): deployments.apps not found",
            "category": "deployment",
            "resolution": "1. Check namespace\n2. Verify deployment yaml\n3. Check kubectl context",
            "severity": "critical",
            "frequency": 0.9
        },
        # Add more k8s errors
    ]
}

@app.route('/encode', methods=['POST'])
def encode():
    try:
        data = request.get_json()
        if not data or 'sentences' not in data:
            return jsonify({'error': 'No sentences provided'}), 400
        
        sentences = data['sentences']
        if not isinstance(sentences, list):
            return jsonify({'error': 'Sentences must be a list'}), 400
        
        # Generate embeddings
        embeddings = model.encode(sentences)
        
        # Convert numpy array to list for JSON serialization
        embeddings_list = embeddings.tolist()
        
        return jsonify({
            'embeddings': embeddings_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/similarity', methods=['POST'])
def similarity():
    try:
        data = request.get_json()
        if not data or 'embeddings' not in data:
            return jsonify({'error': 'No embeddings provided'}), 400
        
        # Get threshold if provided, default to 0.0
        threshold = float(data.get('threshold', 0.0))
        
        # Convert list to numpy array
        embeddings = np.array(data['embeddings'])
        
        # Calculate cosine similarity
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        similarities = cosine_similarity(embeddings)
        
        # Apply threshold if specified
        if threshold > 0:
            similarities = np.where(similarities >= threshold, similarities, 0)
        
        # Convert numpy array to list for JSON serialization
        similarities_list = similarities.tolist()
        
        return jsonify({
            'similarities': similarities_list,
            'threshold': threshold
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def semantic_search():
    try:
        data = request.get_json()
        if not data or 'query' not in data or 'documents' not in data:
            return jsonify({'error': 'Query and documents are required'}), 400
        
        query = data['query']
        documents = data['documents']
        top_k = int(data.get('top_k', 5))
        threshold = float(data.get('threshold', 0.0))
        
        # Encode query and documents
        query_embedding = model.encode(query)
        doc_embeddings = model.encode(documents)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append({
                    'document': documents[idx],
                    'similarity': float(similarities[idx])
                })
        
        return jsonify({
            'query': query,
            'results': results,
            'threshold': threshold
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_process', methods=['POST'])
def batch_process():
    try:
        data = request.get_json()
        if not data or 'batches' not in data:
            return jsonify({'error': 'Batches are required'}), 400
        
        batches = data['batches']
        results = []
        
        for batch in batches:
            start_time = time.time()
            
            # Process each batch
            embeddings = model.encode(batch['sentences'])
            similarities = cosine_similarity(embeddings)
            
            batch_result = {
                'embeddings': embeddings.tolist(),
                'similarities': similarities.tolist(),
                'processing_time': time.time() - start_time
            }
            results.append(batch_result)
        
        return jsonify({
            'results': results,
            'total_batches': len(batches)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify_error', methods=['POST'])
def classify_error():
    try:
        data = request.get_json()
        if not data or 'error_log' not in data:
            return jsonify({'error': 'No error log provided'}), 400
        
        error_log = data['error_log']
        
        # Get embeddings for error log
        error_embedding = model.encode(error_log)
        
        # Classify error type
        error_type_scores = {}
        for error_type, errors in devops_errors.items():
            type_embeddings = model.encode([e["error_log"] for e in errors])
            similarities = cosine_similarity([error_embedding], type_embeddings)[0]
            error_type_scores[error_type] = np.max(similarities)
        
        # Get top error type
        top_error_type = max(error_type_scores.items(), key=lambda x: x[1])[0]
        
        return jsonify({
            'error_type': top_error_type,
            'confidence': float(error_type_scores[top_error_type]),
            'scores': {k: float(v) for k, v in error_type_scores.items()}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/find_solution', methods=['POST'])
def find_solution():
    try:
        data = request.get_json()
        if not data or 'error_log' not in data:
            return jsonify({'error': 'No error log provided'}), 400
        
        error_log = data['error_log']
        error_type = data.get('error_type')  # Optional: pre-classified error type
        
        # If error type not provided, classify it
        if not error_type:
            error_type = classify_error().json['error_type']
        
        # Get embeddings for error log
        error_embedding = model.encode(error_log)
        
        # Find similar errors in the specified category
        similar_errors = []
        for error in devops_errors.get(error_type, []):
            error_emb = model.encode(error["error_log"])
            similarity = cosine_similarity([error_embedding], [error_emb])[0][0]
            similar_errors.append({
                'error_log': error["error_log"],
                'resolution': error["resolution"],
                'similarity': float(similarity),
                'severity': error["severity"],
                'frequency': error["frequency"]
            })
        
        # Sort by similarity and frequency
        similar_errors.sort(key=lambda x: (x['similarity'], x['frequency']), reverse=True)
        
        return jsonify({
            'error_type': error_type,
            'solutions': similar_errors[:3]  # Top 3 solutions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_trends', methods=['POST'])
def analyze_trends():
    try:
        data = request.get_json()
        if not data or 'error_logs' not in data:
            return jsonify({'error': 'No error logs provided'}), 400
        
        error_logs = data['error_logs']
        
        # Analyze error trends
        error_counts = {}
        severity_counts = {}
        resolution_times = []
        
        for error_log in error_logs:
            # Classify error
            error_type = classify_error().json['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            # Get solution and severity
            solution = find_solution().json['solutions'][0]
            severity = solution['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Mock resolution time (in production, this would be real data)
            resolution_times.append(np.random.randint(5, 30))
        
        return jsonify({
            'error_distribution': error_counts,
            'severity_distribution': severity_counts,
            'average_resolution_time': float(np.mean(resolution_times)),
            'common_errors': sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 