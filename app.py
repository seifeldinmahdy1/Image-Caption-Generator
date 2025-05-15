import os
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import pickle
from model import ImageEncoder, TopDownAttention
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']

vocab_size = len(word2idx)
encoder = ImageEncoder(embed_dim=512).to(device)
generator = TopDownAttention(
    embed_dim=512,
    decoder_dim=1024,
    vocab_size=vocab_size,
    word2idx=word2idx,
    encoder_dim=512
).to(device)

def load_models():
    try:
        checkpoint = torch.load('final_model_weights.pth', map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        encoder.eval()
        generator.eval()
        print("Model weights loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return False

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_caption(image_data, beam_size=5):
    """Generate caption for an image from memory"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            encoder_out = encoder(image_tensor)
            seq = generator.sample(encoder_out, beam_size=beam_size)
        
        caption_tokens = []
        for idx in seq:
            idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
            if idx_val not in [word2idx['<pad>'], word2idx['<start>'], word2idx['<end>']]:
                caption_tokens.append(idx2word[idx_val])
        
        return ' '.join(caption_tokens)
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Error generating caption"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        file_data = file.read()
        
        caption = generate_caption(file_data)
        
        return jsonify({
            'success': True,
            'caption': caption
        })

if __name__ == '__main__':
    if load_models():
        app.run(debug=True)
    else:
        print("Failed to load models. Application not started.")
