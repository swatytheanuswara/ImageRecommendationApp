import os
from starlette.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

# ##### CORSHEADERS CONFIGURATION ##############################

CORS_ALLOWED_ORIGINS = ['*']

CORS_ALLOW_METHODS = ['*']

CORS_ALLOW_HEADERS =['*']


middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=CORS_ALLOW_METHODS,
        allow_headers=CORS_ALLOW_HEADERS,
)]


