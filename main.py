from flask import Flask, render_template, request
import base64
import cv2
import numpy as np
import os
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load pre-trained MobileNetV2 model
mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to extract features from images using MobileNetV2
def extract_features(images):
    preprocessed_images = np.array([preprocess_input(cv2.resize(image, (224, 224))) for image in images])
    features = mobilenet_model.predict(preprocessed_images)
    return features.reshape(features.shape[0], -1)

# Function to find k-nearest neighbors
def find_nearest_neighbors(reference_features, image_features, k=5):
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(image_features)
    distances, indices = nn.kneighbors(reference_features)
    return distances, indices

# Function to load dataset images from folders
def load_dataset_from_folders(dataset_folder):
    categories = os.listdir(dataset_folder)
    dataset_images = []
    for category in categories:
        category_folder = os.path.join(dataset_folder, category)
        if os.path.isdir(category_folder):
            images = [cv2.imread(os.path.join(category_folder, file)) for file in os.listdir(category_folder)
                      if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png')]
            dataset_images.extend(images)
    return dataset_images

# Main function
def main(reference_image_path, dataset_folder, top_k=5):
    # Load reference image
    reference_image = cv2.imread(reference_image_path)
    # Extract features from reference image
    reference_features = extract_features([reference_image])
    # Load dataset images from folders
    dataset_images = load_dataset_from_folders(dataset_folder)
    # Extract features from dataset images
    image_features = extract_features(dataset_images)
    # Find k-nearest neighbors
    distances, indices = find_nearest_neighbors(reference_features, image_features, top_k)
    # Return top k closest matches
    matches = []
    for i in range(top_k):
        idx = indices[0][i]
        img = cv2.imencode('.png', dataset_images[idx])[1]
        img_base64 = base64.b64encode(img).decode()
        matches.append(img_base64)
    return matches

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    reference_image = request.files['reference_image']
    reference_image_path = "temp.jpg"  # Temporary path for storing the uploaded image
    reference_image.save(reference_image_path)
    dataset_folder = request.form['dataset_folder']
    top_k = int(request.form['top_k'])
    matches = main(reference_image_path, dataset_folder, top_k)
    return render_template('result.html', matches=matches)


if __name__ == "__main__":
    app.run(debug=True)
