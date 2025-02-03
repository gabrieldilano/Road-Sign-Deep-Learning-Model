from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("Road_Sign_Classification.h5")

# Define classes
class_mapping = ["trafficlight", "stop", "speedlimit", "crosswalk"]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    
    try:
        # Convert the image to array
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Read image using OpenCV

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to match model input
        image_resized = cv2.resize(image, (64, 64))

        # Normalize and add batch dimension
        image_array = np.array(image_resized) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions)  # Get the index of the highest probability
        predicted_class = class_mapping[predicted_class_index]
        confidence = float(np.max(predictions))

        return jsonify({'class': predicted_class, 'confidence': confidence})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
