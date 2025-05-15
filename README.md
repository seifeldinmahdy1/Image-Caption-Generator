# Image-Caption-Generator

A Flask web application that uses a trained image captioning model to generate captions for uploaded images.

## Setup Instructions

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the vocabulary**

   Run the script to create the vocabulary from your trained model:
   
   ```bash
   python save_vocab.py
   ```

   This will create a `vocab.pkl` file needed by the application.

3. **Run the jupyter notebook to get model weights**

   The trained model weights will be saved to this file (`final_model_weights.pth`) in the project directory.

4. **Run the application**

   ```bash
   python app.py
   ```

   The application will be available at http://127.0.0.1:5000/

## Usage

1. Open the web application in your browser
2. Upload an image by dragging and dropping or using the browse button
3. Wait for the model to generate a caption
4. View the generated caption below the image

## Project Structure

- `app.py` - Main Flask application
- `model.py` - Neural network model definitions
- `save_vocab.py` - Script to create vocabulary from COCO dataset
- `templates/` - HTML templates for the web interface
