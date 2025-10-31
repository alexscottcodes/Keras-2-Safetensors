# Quick Start Guide

Get started with the Keras to Safetensors converter in 5 minutes!

## Step 1: Install Cog

**macOS:**
```bash
brew install replicate/tap/cog
```

**Linux/WSL:**
```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog
```

Verify installation:
```bash
cog --version
```

## Step 2: Set Up Your Project

Create a new directory and add the files:

```bash
mkdir my-converter
cd my-converter

# Copy the cog.yaml and predict.py files here
```

Your directory structure should look like:
```
my-converter/
├── cog.yaml
├── predict.py
└── README.md
```

## Step 3: Test Locally

Run a test conversion with a public HuggingFace model:

```bash
cog predict \
  -i huggingface_repo="keras-io/mobilenet-v3-large-100" \
  -i model_type="image_classifier"
```

This will:
1. Download the model from HuggingFace
2. Convert it to Safetensors format
3. Create a zip file with all outputs

## Step 4: Use Your Own Model

Convert your own model hosted on HuggingFace:

```bash
cog predict \
  -i huggingface_repo="YOUR_USERNAME/your-model-name" \
  -i model_type="image_classifier"
```

### If You Have Multiple .h5 Files

Specify which file to convert:

```bash
cog predict \
  -i huggingface_repo="YOUR_USERNAME/your-model-name" \
  -i model_type="image_classifier" \
  -i h5_filename="my_model.h5"
```

## Step 5: Extract and Use the Output

The converter creates a `converted_model.zip` file containing:

```bash
# Extract the zip file
unzip converted_model.zip -d converted_model

# View the contents
ls -la converted_model/
```

You should see:
- `model.safetensors` - Model weights
- `config.json` - Model configuration
- `labels.txt` - Class labels (if available)
- `README.md` - Usage instructions

## Step 6: Load Your Converted Model

Use the provided example script:

```bash
python example_usage.py
```

Or load it manually in Python:

```python
import json
import tensorflow as tf
from safetensors.tensorflow import load_file

# Load config
with open('converted_model/config.json', 'r') as f:
    config = json.load(f)

# Recreate model
model = tf.keras.models.model_from_json(
    json.dumps(config['architecture'])
)

# Load weights
weights = load_file('converted_model/model.safetensors')

# Apply weights to model
for layer in model.layers:
    layer_weights = []
    for i in range(len(layer.get_weights())):
        key = f"{layer.name}.weight_{i}"
        if key in weights:
            layer_weights.append(weights[key])
    if layer_weights:
        layer.set_weights(layer_weights)

print("Model ready!")
```

## Common Model Types

### Image Classifier
```bash
cog predict \
  -i huggingface_repo="username/image-model" \
  -i model_type="image_classifier"
```

### Sound Classifier
```bash
cog predict \
  -i huggingface_repo="username/audio-model" \
  -i model_type="sound_classifier"
```

### Text Classifier
```bash
cog predict \
  -i huggingface_repo="username/text-model" \
  -i model_type="text_classifier"
```

## Troubleshooting

### Issue: "Repository not found"
**Solution:** Check your repository ID format: `username/repo-name`

### Issue: "No .h5 file found"
**Solution:** Specify the filename with `-i h5_filename="your_model.h5"`

### Issue: Cog command not found
**Solution:** Make sure Cog is in your PATH:
```bash
which cog
```

### Issue: Docker errors
**Solution:** Ensure Docker is running:
```bash
docker info
```

## Next Steps

- **Deploy to Replicate**: See the main README for deployment instructions
- **Customize**: Modify `predict.py` to add custom preprocessing
- **Batch Convert**: Create a script to convert multiple models

## Example Projects

Here are some example HuggingFace repositories you can try:

1. **Image Classification:**
   - `keras-io/mobilenet-v3-large-100`
   - `tensorflow/efficientnet-b0`

2. **Custom Models:**
   - Upload your own `.h5` model to HuggingFace
   - Convert it using this tool

## Getting Help

- 📖 Full documentation: See `README.md`
- 💬 Cog documentation: https://github.com/replicate/cog
- 🤗 HuggingFace docs: https://huggingface.co/docs
- 🔐 Safetensors info: https://github.com/huggingface/safetensors

## Tips for Success

1. **Start Simple**: Test with a small model first
2. **Check Compatibility**: Ensure your model is in Keras `.h5` format
3. **Verify Outputs**: Always check the generated `README.md` in the zip
4. **Test Thoroughly**: Load the converted model and verify predictions
5. **Keep Organized**: Store original and converted models separately

## Performance Notes

- **Small models** (<100MB): Convert in seconds
- **Medium models** (100MB-1GB): Convert in 1-2 minutes
- **Large models** (>1GB): May take several minutes

Conversion time depends on:
- Model size
- Internet speed (for downloading)
- Available RAM
- CPU/GPU availability

---

**Ready to convert?** Run your first conversion now:

```bash
cog predict \
  -i huggingface_repo="YOUR_REPO" \
  -i model_type="image_classifier"
```

Happy converting! 🚀