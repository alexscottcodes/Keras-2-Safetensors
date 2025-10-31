# Keras to Safetensors Converter (Cog)

A Replicate Cog script for converting Keras `.h5` models to the Safetensors format. This tool downloads models from HuggingFace repositories, converts them to the safer and more efficient Safetensors format, and packages everything with proper configuration files.

## Features

- 🔄 **Automatic Download**: Downloads models directly from HuggingFace repositories
- 🔐 **Safe Format**: Converts to Safetensors format (no pickle security risks)
- 📦 **Complete Package**: Includes weights, architecture config, and labels
- 🎯 **Classifier Support**: Supports image, sound, and text classifiers
- 📝 **Documentation**: Auto-generates README with usage instructions

## Prerequisites

- Docker (for running Cog)
- Cog installed ([installation guide](https://github.com/replicate/cog))
- A HuggingFace repository with a Keras `.h5` model

## Installation

1. Install Cog:
```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog
```

2. Clone this repository or create the files:
```bash
mkdir keras-to-safetensors
cd keras-to-safetensors
# Copy the cog.yaml and predict.py files here
```

## Usage

### Local Testing

Test the conversion locally with Cog:

```bash
cog predict \
  -i huggingface_repo="username/model-name" \
  -i model_type="image_classifier"
```

#### Parameters

- `huggingface_repo` (required): HuggingFace repository ID (e.g., "username/model-name")
- `model_type` (required): Type of classifier
  - `image_classifier`
  - `sound_classifier`
  - `text_classifier`
- `h5_filename` (optional): Specific .h5 file to convert (if multiple exist)

### Example with Specific Model File

```bash
cog predict \
  -i huggingface_repo="username/my-classifier" \
  -i model_type="image_classifier" \
  -i h5_filename="model.h5"
```

### Building the Docker Image

Build the Docker image for deployment:

```bash
cog build -t keras-to-safetensors
```

### Running the HTTP Server

Start a local HTTP server for testing:

```bash
cog run -p 5000 python -m cog.server.http
```

Then make predictions via HTTP:

```bash
curl http://localhost:5000/predictions -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "huggingface_repo": "username/model-name",
      "model_type": "image_classifier"
    }
  }'
```

## Pushing to Replicate

1. Create a model on Replicate:
   - Visit [replicate.com/create](https://replicate.com/create)
   - Choose a name for your model

2. Update `cog.yaml` with your model info:
```yaml
image: "r8.im/your-username/keras-to-safetensors"
```

3. Push to Replicate:
```bash
cog login
cog push r8.im/your-username/keras-to-safetensors
```

## Output

The converter creates a zip file containing:

1. **model.safetensors**: Model weights in Safetensors format
2. **config.json**: Complete model architecture and metadata
3. **labels.txt**: Class labels (if found in the repository)
4. **README.md**: Usage instructions for loading the converted model

### Example Output Structure

```
converted_model.zip
├── model.safetensors
├── config.json
├── labels.txt
└── README.md
```

## Loading Converted Models

### Python Example

```python
import tensorflow as tf
import json
from safetensors.tensorflow import load_file

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Recreate model architecture
model = tf.keras.models.model_from_json(
    json.dumps(config['architecture'])
)

# Load and apply weights
weights_dict = load_file('model.safetensors')

for layer in model.layers:
    layer_weights = []
    for i in range(len(layer.get_weights())):
        key = f"{layer.name}.weight_{i}"
        if key in weights_dict:
            layer_weights.append(weights_dict[key])
    if layer_weights:
        layer.set_weights(layer_weights)

# Load labels
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

print(f"Model loaded with {len(labels)} classes")
```

## How It Works

1. **Download**: Uses `huggingface_hub` to download the entire repository
2. **Find Model**: Locates the `.h5` file (or uses specified filename)
3. **Load**: Loads the Keras model with TensorFlow
4. **Extract**: Extracts all layer weights into a dictionary
5. **Convert**: Saves weights in Safetensors format
6. **Package**: Creates config JSON and copies labels
7. **Zip**: Bundles everything into a single archive

## Benefits of Safetensors

According to research, Safetensors offers several advantages:

- **Security**: No code execution risks (unlike pickle)
- **Speed**: Faster loading times with zero-copy reads
- **Efficiency**: Optimized for large tensors
- **Compatibility**: Works across different frameworks
- **Transparency**: Simple, inspectable format

## Technical Details

### Dependencies

- **TensorFlow 2.15.0**: For loading Keras models
- **Safetensors 0.4.2**: For safe tensor storage
- **HuggingFace Hub 0.20.3**: For downloading models
- **NumPy 1.26.3**: For array operations

### Model Architecture Preservation

The converter preserves:
- Complete layer architecture
- Input/output shapes
- Layer configurations
- Total parameter count
- Number of classes (for classifiers)

### Weight Naming Convention

Weights are stored with keys: `{layer_name}.weight_{index}`

Example:
```
conv2d_1.weight_0  # First weight tensor of conv2d_1 layer
conv2d_1.weight_1  # Second weight tensor (bias)
dense_2.weight_0   # First weight tensor of dense_2 layer
```

## Troubleshooting

### "No .h5 file found"
- Specify the exact filename with `h5_filename` parameter
- Ensure the HuggingFace repo actually contains a Keras model

### "Repository not found"
- Check the repository ID format: `username/repo-name`
- Verify the repository is public or you have access

### Memory Issues
- Large models may require more RAM
- Consider using GPU-enabled Cog for very large models

## Contributing

Contributions are welcome! Areas for improvement:

- Support for more model formats (SavedModel, ONNX)
- Batch conversion of multiple models
- Model validation after conversion
- Support for custom layer types

## License

This project is provided as-is for educational and practical use.

## Acknowledgments

- Based on [Cog](https://github.com/replicate/cog) by Replicate
- Uses [Safetensors](https://github.com/huggingface/safetensors) by HuggingFace
- Integrates with [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub)

## References

- [Safetensors Documentation](https://huggingface.co/docs/safetensors)
- [Cog Documentation](https://github.com/replicate/cog/blob/main/docs/getting-started.md)
- [HuggingFace Hub Guide](https://huggingface.co/docs/huggingface_hub/guides/download)
- [Keras Model Serialization](https://www.tensorflow.org/guide/keras/serialization_and_saving)