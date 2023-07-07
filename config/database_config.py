import os
from config import env_config

DATABASE_CLIENT = os.environ.get("DATABASE_CLIENT")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
DATABASE_COLLECTION = os.environ.get("DATABASE_COLLECTION")