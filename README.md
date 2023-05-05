# wiser-image-recommendation-app
This is a FastAPI application designed to provide image recommendations using Deep Learning algorithms.

## Features
### Image-to-Image Recommendation
Users can upload an image, and the application will use advanced Deep Learning algorithms to recommend similar images based on the uploaded one. This feature is helpful for users looking for visually similar images for various purposes.

### Text-to-Image Recommendation
Users can input text descriptions, and the application will generate and recommend images that best match the provided text. This feature can be particularly useful for users seeking images based on specific descriptions or keywords.

## Tech stack
- Python 3.10.6
- FastAPI
- TensorFlow
- PyTorch
- NVIDIA Driver: 525.x
- Cuda 11.8
- CuDNN 11.8 compatible
- Ubuntu 22.04 LTS (Tested)
- MongoDB

## Setup and Installation
 1) Navigate the project directory
 2) Install the required dependencies: ```pip install -r requirements.txt```
 3) Copy .env file (**Restricted for the public acess as it has the sensitive information like keys and credentials, so do not share without permission.**)
 4) Run the FastAPI application: ```python asgi.py```

## Usage
Once the application is up and running, you can interact with it through HTTP requests or through a web interface in localhost.

### Endpoints
- `/image/inference`: Endpoint for storing features of images.
- `/search/image`: Endpoint for image-to-image recommendation.
- `/search/text`: Endpoint for text-to-image recommendation.
- ***Note:*** Check API documentation for the more information

## Technical Information
- Image captioning model: **Blip Caption**
- Image features extract model: **ResNet50**
- Text embedding model: **Sentence Transformer BERT**