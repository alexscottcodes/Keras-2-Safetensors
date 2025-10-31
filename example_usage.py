"""
Example usage script for the converted Safetensors model.
This demonstrates how to load and use a model that has been converted
from Keras .h5 format to Safetensors.
"""

import json
import numpy as np
import tensorflow as tf
from safetensors.tensorflow import load_file
from pathlib import Path


def load_converted_model(model_dir: str):
    """
    Load a model that has been converted to Safetensors format.
    
    Args:
        model_dir: Directory containing the converted model files
        
    Returns:
        Tuple of (model, labels, config)
    """
    model_path = Path(model_dir)
    
    # Load configuration
    print("📖 Loading model configuration...")
    with open(model_path / 'config.json', 'r') as f:
        config = json.load(f)
    
    print(f"   Model type: {config['model_type']}")
    print(f"   Total parameters: {config['total_parameters']:,}")
    print(f"   Number of classes: {config.get('num_classes', 'N/A')}")
    
    # Recreate model architecture from JSON
    print("🔨 Reconstructing model architecture...")
    model = tf.keras.models.model_from_json(
        json.dumps(config['architecture'])
    )
    
    # Load weights from Safetensors
    print("⚖️  Loading weights from Safetensors...")
    weights_dict = load_file(str(model_path / 'model.safetensors'))
    
    # Apply weights to model layers
    print("🔄 Applying weights to layers...")
    for layer in model.layers:
        layer_weights = []
        for i in range(len(layer.get_weights())):
            key = f"{layer.name}.weight_{i}"
            if key in weights_dict:
                layer_weights.append(weights_dict[key])
        if layer_weights:
            layer.set_weights(layer_weights)
    
    # Load labels if available
    labels = None
    labels_file = model_path / 'labels.txt'
    if labels_file.exists():
        print("🏷️  Loading class labels...")
        with open(labels_file, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        print(f"   Loaded {len(labels)} labels")
    else:
        print("⚠️  No labels file found")
    
    print("✅ Model loaded successfully!")
    return model, labels, config


def predict_image_classifier(model, image_path: str, labels: list, 
                            input_shape: tuple = (224, 224)):
    """
    Make a prediction with an image classifier.
    
    Args:
        model: Loaded Keras model
        image_path: Path to input image
        labels: List of class labels
        input_shape: Expected input shape (height, width)
        
    Returns:
        Dictionary with predictions
    """
    from PIL import Image
    
    print(f"\n🖼️  Processing image: {image_path}")
    
    # Load and preprocess image
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(input_shape)
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    print("🔮 Making prediction...")
    predictions = model.predict(img_array, verbose=0)
    
    # Get top 5 predictions
    top_indices = np.argsort(predictions[0])[-5:][::-1]
    
    results = {
        'predictions': []
    }
    
    print("\n📊 Top 5 Predictions:")
    print("-" * 50)
    for i, idx in enumerate(top_indices, 1):
        label = labels[idx] if labels and idx < len(labels) else f"Class {idx}"
        confidence = float(predictions[0][idx])
        print(f"{i}. {label:30s} {confidence:6.2%}")
        
        results['predictions'].append({
            'rank': i,
            'class_index': int(idx),
            'label': label,
            'confidence': confidence
        })
    
    return results


def predict_sound_classifier(model, audio_path: str, labels: list):
    """
    Make a prediction with a sound classifier.
    
    Args:
        model: Loaded Keras model
        audio_path: Path to input audio file
        labels: List of class labels
        
    Returns:
        Dictionary with predictions
    """
    print(f"\n🔊 Processing audio: {audio_path}")
    print("⚠️  Note: Implement audio preprocessing based on your model's requirements")
    
    # This is a placeholder - implement based on your specific model
    # Common approaches:
    # - Load audio with librosa
    # - Extract mel spectrograms or MFCCs
    # - Normalize and reshape for model input
    
    # Example structure:
    # import librosa
    # audio, sr = librosa.load(audio_path, sr=16000)
    # features = extract_features(audio)
    # predictions = model.predict(features)
    
    return {'message': 'Implement audio preprocessing for your specific model'}


def predict_text_classifier(model, text: str, labels: list, 
                           max_length: int = 512):
    """
    Make a prediction with a text classifier.
    
    Args:
        model: Loaded Keras model
        text: Input text to classify
        labels: List of class labels
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with predictions
    """
    print(f"\n📝 Processing text: {text[:100]}...")
    print("⚠️  Note: Implement text preprocessing based on your model's requirements")
    
    # This is a placeholder - implement based on your specific model
    # Common approaches:
    # - Tokenize text
    # - Convert to token IDs
    # - Pad/truncate to max_length
    # - Make prediction
    
    # Example structure:
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(...)
    # inputs = tokenizer(text, max_length=max_length, ...)
    # predictions = model.predict(inputs)
    
    return {'message': 'Implement text preprocessing for your specific model'}


def main():
    """Example usage of the converted model."""
    
    # Example 1: Load the model
    print("=" * 60)
    print("EXAMPLE: Loading Converted Safetensors Model")
    print("=" * 60)
    
    model_dir = "./converted_model"  # Update with your path
    
    try:
        model, labels, config = load_converted_model(model_dir)
        
        # Print model summary
        print("\n📋 Model Summary:")
        print("-" * 60)
        model.summary()
        
        # Example 2: Make predictions based on model type
        model_type = config['model_type']
        
        if model_type == 'image_classifier':
            print("\n" + "=" * 60)
            print("EXAMPLE: Image Classification")
            print("=" * 60)
            
            # Replace with your actual image path
            image_path = "example_image.jpg"
            
            # Uncomment to run prediction:
            # results = predict_image_classifier(
            #     model, image_path, labels, 
            #     input_shape=(224, 224)
            # )
            # print(f"\n✅ Prediction complete: {results['predictions'][0]['label']}")
            
            print("\n💡 To make predictions, provide an image path and uncomment the code above")
            
        elif model_type == 'sound_classifier':
            print("\n" + "=" * 60)
            print("EXAMPLE: Sound Classification")
            print("=" * 60)
            print("💡 Implement audio preprocessing based on your model's requirements")
            
        elif model_type == 'text_classifier':
            print("\n" + "=" * 60)
            print("EXAMPLE: Text Classification")
            print("=" * 60)
            print("💡 Implement text preprocessing based on your model's requirements")
        
        # Example 3: Save as standard Keras model (optional)
        print("\n" + "=" * 60)
        print("OPTIONAL: Save as Standard Keras Model")
        print("=" * 60)
        
        # Uncomment to save:
        # output_path = "./restored_keras_model.h5"
        # model.save(output_path)
        # print(f"✅ Model saved to {output_path}")
        
        print("\n💡 To save the model, uncomment the code above")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Model directory not found: {model_dir}")
        print("\n💡 Make sure to:")
        print("   1. Extract the converted_model.zip file")
        print("   2. Update the model_dir path in this script")
    except Exception as e:
        print(f"\n❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()