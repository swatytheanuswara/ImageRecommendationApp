import os
import torch
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor


from apps.utils.aws import load_image_from_s3

from config import env_config

# BLIP image captioning model loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lavis_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

# Load Spacy model
# try:
#     nlp = spacy.load("en_core_web_sm")

# except Exception as e:
#     print("Spacy model can't found")
#     print("Downloading spacy model...")
    
#     import subprocess
#     command = "python -m spacy download en_core_web_sm"
#     subprocess.run(command, shell=True)
#     print("Spacy model downloaded ...")
#     nlp = spacy.load("en_core_web_sm")

# Load sentence BERT model
sentence_bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Load Model
model = resnet50.ResNet50(weights='imagenet')

# Feature extractor layer
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)


def extract_image_features(imagepath):
    local_image_path = load_image_from_s3(imagepath)
    original_img = load_img(local_image_path, target_size=(224, 224))
    np_img = img_to_array(original_img)
    preprocess_img = preprocess_input(np_img)
    img_features = feat_extractor.predict(np.expand_dims(preprocess_img, axis=0)).tolist()
    output_sentence, output_sentence_embeddings = extract_text_embeddings(local_image_path)

    return (local_image_path, img_features, output_sentence, output_sentence_embeddings)

def image_search(imagepath, collection):
    local_image_path = load_image_from_s3(imagepath)
    original_img = load_img(local_image_path, target_size=(224, 224))
    np_img = img_to_array(original_img)
    preprocess_img = preprocess_input(np_img)
    img_features = feat_extractor.predict(np.expand_dims(preprocess_img, axis=0))
    
    # Retrieve all image features from MongoDB
    stored_features = [doc['features'] for doc in collection.find({}, {'features': 1})]
    # import pdb; pdb.set_trace()
    # Calculate cosine similarity using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        similarities = list(executor.map(lambda x: cosine_similarity(img_features, np.array([x]))[0][0], stored_features))
    
    # Sort the similarities in descending order
    sorted_indices = np.argsort(similarities)[::-1]

    batch_size = 100  # Adjust the batch size as needed
    batch_indices = sorted_indices[:batch_size]  # Take the first batch
    batch_stored_features = [stored_features[idx] for idx in batch_indices]

    batch_images = collection.find({'features': {'$in': batch_stored_features}})
    similar_images = {str(image['features']): image for image in batch_images}
    sorted_similar_images = [similar_images[str(feature)] for feature in batch_stored_features]


    # Retrieve similar images from MongoDB based on sorted indices
    # similar_images = [collection.find_one({'features': stored_features[idx]}) for idx in sorted_indices]

    return sorted_similar_images[:15]



def extract_text_embeddings(imagepath):
    """
    this function is use for get the context of the image and extract the keyword from the context

    :arg:
        imagepath :- input image path

    :return:
        output_sentence : str - Sentence text
        output_sentence_embeddings: list - list of embeddings of sentences
        
    """

    # load sample image
    raw_image = Image.open(imagepath).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # generate caption
    output_sentence = lavis_model.generate({"image": image})[0]
    output_sentence_embeddings = sentence_bert_model.encode(output_sentence)

    return output_sentence, output_sentence_embeddings


def text_search(context, collection) -> list:
    """
    This function takes the context of the image and extracts keywords to find similar images based on those keywords.

    :param:
        context: The context (description) of the image.
        collection: mongo collection

    :return: 
        A list of dictionaries containing "product_id" and "image_name" keys for matching images.
    """
    
    # Set threhsold
    threshold = float(os.environ.get('SENTENCE_THRESHOLD'))

    # List all the embeddings
    all_embeddings = [emb['sentence_embeddings'] for emb in collection.find({}, {'product_id': 1, 'image_s3_path': 1, 'sentence_embeddings': 1})]

    # Cosine similarity check
    output_sentence_embeddings = sentence_bert_model.encode(context).tolist()
    similarities = cosine_similarity([output_sentence_embeddings], all_embeddings)

    # Find the indices of similar sentences that pass the threshold
    similar_sentences_indices = np.where(similarities[0] >= threshold)[0]

    # Retrieve unique images from DB
    matching_images = []
    seen_image_names = set()  # Set to keep track of unique image names

    for idx in similar_sentences_indices:
        product_data = collection.find_one({'sentence_embeddings': all_embeddings[idx]})
        image_name = product_data['image_s3_path']
        
        if image_name not in seen_image_names:  # Check if the image name is not already seen
            matching_images.append({"product_id": product_data['product_id'], "image_name": image_name})
            seen_image_names.add(image_name)  # Add the image name to the set
    
    return matching_images[:15]