import os
import boto3
from config import aws_config

# Get the bucket and base URL for s3
bucket_name = aws_config.S3_BUCKET
base_url = aws_config.AWS_BASE_URL
access_key = aws_config.ACCESS_KEY
secret_key = aws_config.SECRET_KEY

# Create an S3 client
s3 = boto3.client('s3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key)


def load_image_from_s3(image_key):
    # Download the image file from S3
    local_dir = os.path.join("data")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    local_image_path = os.path.join(local_dir, os.path.basename(image_key))
    s3.download_file(bucket_name, image_key, local_image_path)

    return local_image_path