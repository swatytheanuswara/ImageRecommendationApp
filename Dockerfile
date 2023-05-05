# Use python 3.10-alpine light python base image for testing
FROM python:3.10

# Set myproject as working directory and add all files into it
RUN mkdir /wiser-image-recommendation-app
WORKDIR /wiser-image-recommendation-app

ADD . /wiser-image-recommendation-app

# Upgrade pip and install requirements
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Run wsgi file
CMD ["python", "run.py"]

EXPOSE 8000