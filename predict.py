"""
Cog predictor for converting Keras .h5 models to Safetensors format.
Downloads models from HuggingFace repositories and converts them.
"""

import os
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
import shutil

import tensorflow as tf
import numpy as np
from safetensors.tensorflow import save_file as save_file_tf
from huggingface_hub import snapshot_download, hf_hub_download
from cog import BasePredictor, Input, Path as CogPath


class Predictor(BasePredictor):
    """
    Predictor for converting Keras .h5 models to Safetensors format.
    """

    def predict(
        self,
        huggingface_repo: str = Input(
            description="HuggingFace repository ID (e.g., 'username/model-name')",
            default=None
        ),
        model_type: str = Input(
            description="Type of classifier model",
            choices=["image_classifier", "sound_classifier", "text_classifier"],
            default="image_classifier"
        ),
        h5_filename: str = Input(
            description="Name of the .h5 model file in the repository (if not provided, will search for .h5 files)",
            default=None
        ),
    ) -> CogPath:
        """
        Convert a Keras .h5 model from HuggingFace to Safetensors format.
        
        Args:
            huggingface_repo: HuggingFace repository ID
            model_type: Type of classifier model
            h5_filename: Optional specific .h5 filename to convert
            
        Returns:
            Path to a zip file containing the converted model
        """
        
        if not huggingface_repo:
            raise ValueError("huggingface_repo is required")
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        download_dir = os.path.join(temp_dir, "downloaded")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            print(f"📥 Downloading repository: {huggingface_repo}")
            
            # Download the entire repository from HuggingFace
            repo_path = snapshot_download(
                repo_id=huggingface_repo,
                local_dir=download_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"✓ Repository downloaded to: {repo_path}")
            
            # Find .h5 model file
            h5_file = self._find_h5_file(repo_path, h5_filename)
            
            if not h5_file:
                raise FileNotFoundError(
                    f"No .h5 file found in repository. "
                    f"Please specify the h5_filename parameter."
                )
            
            print(f"📦 Found model file: {os.path.basename(h5_file)}")
            
            # Load the Keras model
            print("🔄 Loading Keras model...")
            model = tf.keras.models.load_model(h5_file, compile=False)
            print(f"✓ Model loaded successfully")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Total parameters: {model.count_params():,}")
            
            # Extract model weights and convert to safetensors
            print("🔄 Converting weights to Safetensors format...")
            weights_dict = self._extract_weights(model)
            
            safetensors_path = os.path.join(output_dir, "model.safetensors")
            save_file_tf(weights_dict, safetensors_path)
            print(f"✓ Weights saved to model.safetensors")
            
            # Save model architecture as JSON
            print("💾 Saving model configuration...")
            model_config = {
                "model_type": model_type,
                "architecture": json.loads(model.to_json()),
                "input_shape": [
                    [dim if dim is not None else -1 for dim in layer.shape]
                    for layer in model.inputs
                ],
                "output_shape": [
                    [dim if dim is not None else -1 for dim in layer.shape]
                    for layer in model.outputs
                ],
                "num_classes": self._get_num_classes(model),
                "total_parameters": int(model.count_params())
            }
            
            config_path = os.path.join(output_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            print("✓ Configuration saved to config.json")
            
            # Look for labels.txt in the repository
            labels_file = self._find_labels_file(repo_path)
            if labels_file:
                print(f"📝 Found labels file: {os.path.basename(labels_file)}")
                shutil.copy(labels_file, os.path.join(output_dir, "labels.txt"))
                print("✓ Labels copied to output")
            else:
                print("⚠️  No labels.txt file found in repository")
            
            # Create README with usage instructions
            self._create_readme(output_dir, model_config, huggingface_repo)
            
            # Create a zip file with all outputs
            print("📦 Creating output archive...")
            zip_path = os.path.join(temp_dir, "converted_model.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)
            
            print("✅ Conversion complete!")
            print(f"📦 Output archive size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")
            
            return CogPath(zip_path)
            
        except Exception as e:
            print(f"❌ Error during conversion: {str(e)}")
            raise
        
        finally:
            # Cleanup is handled by Cog after returning the file
            pass
    
    def _find_h5_file(self, repo_path: str, h5_filename: Optional[str] = None) -> Optional[str]:
        """Find .h5 file in the repository."""
        if h5_filename:
            # Look for specific file
            h5_path = os.path.join(repo_path, h5_filename)
            if os.path.exists(h5_path):
                return h5_path
            return None
        
        # Search for any .h5 file
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.h5') or file.endswith('.hdf5'):
                    return os.path.join(root, file)
        return None
    
    def _find_labels_file(self, repo_path: str) -> Optional[str]:
        """Find labels.txt file in the repository."""
        common_names = ['labels.txt', 'class_names.txt', 'classes.txt', 'labels.json']
        
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.lower() in [name.lower() for name in common_names]:
                    return os.path.join(root, file)
        return None
    
    def _extract_weights(self, model: tf.keras.Model) -> dict:
        """Extract weights from Keras model into a dictionary."""
        weights_dict = {}
        
        for layer in model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:
                for i, weight in enumerate(layer_weights):
                    # Create a unique key for each weight tensor
                    key = f"{layer.name}.weight_{i}"
                    weights_dict[key] = weight
        
        return weights_dict
    
    def _get_num_classes(self, model: tf.keras.Model) -> Optional[int]:
        """Try to determine the number of output classes."""
        try:
            output_shape = model.output_shape
            if isinstance(output_shape, tuple):
                # Single output
                return output_shape[-1]
            elif isinstance(output_shape, list):
                # Multiple outputs - return first
                return output_shape[0][-1]
        except:
            return None
    
    def _create_readme(self, output_dir: str, model_config: dict, repo_id: str):
        """Create a README file with usage instructions."""
        readme_content = f"""# Converted Keras Model to Safetensors

## Model Information

- **Source Repository**: {repo_id}
- **Model Type**: {model_config['model_type']}
- **Total Parameters**: {model_config['total_parameters']:,}
- **Number of Classes**: {model_config.get('num_classes', 'N/A')}

## Files Included

- `model.safetensors`: Model weights in Safetensors format
- `config.json`: Model architecture and configuration
- `labels.txt`: Class labels (if available)
- `README.md`: This file

## Loading the Model

### Loading Weights into Keras

```python
import tensorflow as tf
import json
from safetensors.tensorflow import load_file

# Load the model configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Recreate the model architecture from config
model = tf.keras.models.model_from_json(
    json.dumps(config['architecture'])
)

# Load weights from safetensors
weights_dict = load_file('model.safetensors')

# Apply weights to model layers
for layer in model.layers:
    layer_weights = []
    for i in range(len(layer.get_weights())):
        key = f"{{layer.name}}.weight_{{i}}"
        if key in weights_dict:
            layer_weights.append(weights_dict[key])
    if layer_weights:
        layer.set_weights(layer_weights)

print("Model loaded successfully!")
```

### Loading Class Labels

```python
# Load labels if available
try:
    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"Loaded {{len(labels)}} class labels")
except FileNotFoundError:
    print("No labels file found")
```

## Model Details

```json
{json.dumps(model_config, indent=2)}
```

## Notes

- The Safetensors format is safer and more efficient than pickle-based formats
- Model architecture is stored separately from weights
- This conversion maintains the original model's weights and structure
- You'll need TensorFlow and safetensors packages to load this model

## Requirements

```
tensorflow>=2.15.0
safetensors>=0.4.0
```

## Conversion Details

This model was converted using a Cog-based converter that:
1. Downloads the model from HuggingFace
2. Loads the Keras .h5 model
3. Extracts weights to Safetensors format
4. Saves model configuration as JSON
5. Preserves class labels if available

For more information about Safetensors: https://github.com/huggingface/safetensors
"""
        
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)