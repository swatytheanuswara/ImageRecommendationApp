import uuid
from typing import List
from fastapi import APIRouter,status
from fastapi.encoders import jsonable_encoder

from apps.models.model import ImageRecommendation
from apps.database import collection
from apps.ml_backend.inference import extract_image_features, image_search, text_search
from apps.utils.standard_response import StandardResponse
from apps import constant


# Specifies route to create APi's
router = APIRouter()


@router.post("/image/inference")
def create_inference(image_recs: List[ImageRecommendation]):
    """
    This API is for creating an image object with features of images.
    """
    try:
        # Convert the pydantic model to a dictionary
        image_obj_dicts = [image_rec.dict() for image_rec in image_recs]
        # Retrieve existing 'image_s3_path' values from collection
        existing_path = collection.distinct("image_s3_path")
        # check each image_s3_path in list
        response = []
        for image_obj in image_obj_dicts:
            image_s3_path = image_obj["image_s3_path"]
            for image in image_s3_path:
                if image in existing_path:
                    image_obj['image_s3_path'] = image
                    error_dict = {"image_s3_path": image, "message": constant.ImageAlreadyExists}
                    response.append(error_dict)
                else:
                    extracted_features = extract_image_features(imagepath=image)
                    features = sum(extracted_features[1], [])
                    image_obj["_id"] = str(uuid.uuid4())
                    image_obj['features'] = features
                    image_obj['image_s3_path'] = image
                    image_obj['sentence'] = extracted_features[2]
                    image_obj['sentence_embeddings'] = extracted_features[3].tolist()
  
                    # Update the document with the extracted features
                    updated_document = collection.insert_one(image_obj)
                    # Retrieve the updated document
                    updated_doc = collection.find_one({'_id': updated_document.inserted_id})
                    updated_data = jsonable_encoder(updated_doc, exclude={"_id", "features", "sentence", "sentence_embeddings"})
                    # Convert ObjectId to string if needed
                    if "_id" in updated_data:
                        updated_data["_id"] = str(updated_data["_id"])
 
                    response.append(updated_data)
                    # Convert ObjectId to string if needed
                    # updated_doc["_id"] = str(updated_doc["_id"])
                    # response.append(updated_doc)

        return StandardResponse(True, status.HTTP_201_CREATED, constant.ImageObjectCreated, response)
    
    except Exception as e:
        return StandardResponse(False, status.HTTP_400_BAD_REQUEST, constant.ErrorOccured, [])
  

@router.get("/search/text")
def search_text_from_collection(query: str):
    """
    This API will return a list of images that match the keywords
    """
    try:
        # import pdb;pdb.set_trace()  
        matching_images = text_search(query, collection)
        
        if matching_images:
            return StandardResponse(True, status.HTTP_200_OK, constant.ImageRetrieved, matching_images)
        else:
            return StandardResponse(True, status.HTTP_200_OK, constant.ImageNotFound, [])
    except Exception as e:
        return StandardResponse(False, status.HTTP_400_BAD_REQUEST, constant.ErrorOccured, [])


@router.get("/search/image")
def search_image_from_collection(img_path: str):
    """This API will return a list of images
    that match the image"""
    try:
        # check if images with same image path exists in collection
        # using image_search function
        response = []
        similar_images = image_search(img_path, collection)
        if similar_images:
            # Convert ObjectId to string for each document
            for image in similar_images:  
                image["_id"] = str(image["_id"])
                data = jsonable_encoder(image, exclude={"_id", "features", "sentence", "sentence_embeddings"})
                response.append(data)     
            return StandardResponse(True, status.HTTP_200_OK, constant.ImageRetrieved, response)
        else:
            return StandardResponse(True, status.HTTP_200_OK, constant.ImageNotFound, [])
    except Exception as e:
        return StandardResponse(False, status.HTTP_400_BAD_REQUEST, constant.ErrorOccured, [])
        
