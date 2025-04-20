from sentence_transformers import SentenceTransformer
from typing import Dict, Any

class ModelManager:
    """Manages sentence transformer models and their configurations"""

    def __init__(self):
        # Local folder containing paraphrase-MiniLM-L6-v2 files
        local_cache_dir = r"C:\Users\krish\.cache\torch\sentence_transformers\sentence-transformers_paraphrase-MiniLM-L6-v2"

        # Available models and their local paths
        self.available_models: Dict[str, Dict[str, Any]] = {
            "paraphrase-MiniLM-L6-v2": {
                "description": "Fast and efficient model for general purpose embeddings",
                "max_seq_length": 128,
                "dimensions": 384,
                "path": local_cache_dir
            },
            "all-MiniLM-L6-v2": {
                "description": "General purpose model trained on a large dataset",
                "max_seq_length": 128,
                "dimensions": 384,
                "path": "all-MiniLM-L6-v2"
            },
            "all-mpnet-base-v2": {
                "description": "High quality model with better performance but slower",
                "max_seq_length": 128,
                "dimensions": 768,
                "path": "all-mpnet-base-v2"
            }
        }

        # Load default model from the specified local directory, enforce offline mode
        default_entry = self.available_models["paraphrase-MiniLM-L6-v2"]
        model_path = default_entry["path"]
        self.current_model = SentenceTransformer(model_path, local_files_only=True)

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Return metadata for each model (excluding internal paths)"""
        return {
            name: {k: v for k, v in info.items() if k != "path"}
            for name, info in self.available_models.items()
        }

    def get_current_model(self) -> SentenceTransformer:
        """Return the currently loaded model"""
        return self.current_model

    def get_current_model_info(self) -> Dict[str, Any]:
        """Return info about the currently active model"""
        model_name = self.current_model.get_name()
        info = self.available_models.get(model_name, {})
        return {
            "name": model_name,
            "info": {k: v for k, v in info.items() if k != "path"}
        }

    def switch_model(self, model_name: str) -> Dict[str, Any]:
        """Switch to a different model, loading from its configured path and enforcing offline mode"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")

        entry = self.available_models[model_name]
        model_path = entry.get("path", model_name)
        self.current_model = SentenceTransformer(model_path, local_files_only=True)

        return {
            "message": f"Switched to model {model_name}",
            "model_info": {k: v for k, v in entry.items() if k != "path"}
        }

# Global instance will load from the local cache directory with offline enforcement
model_manager = ModelManager()
