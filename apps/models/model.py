from pydantic import BaseModel
from typing import List, Optional

class ImageRecommendation(BaseModel):
    product_id: str
    image_s3_path: List[str]
    features: Optional[List[float]] = []
    # keywords: Optional[List[str]] = []
    sentence: Optional[str]
    sentence_embeddings: Optional[List[float]] = []

    class Config:
        schema_extra = {
            "example": {
                "product_id": "12345",
                "image_s3_path": "https://art-image-bucket.s3.amazonaws.com/artifacts/image.jpg",
                "features": ["feature1", "feature2"],
                "sentence": "Testing sentence",
                "sentence_embeddings": [1.0, 2.0, 3.0]
                # "keywords": ["keyword1", "keyword2"]
            }
        }