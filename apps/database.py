from pymongo import MongoClient
from config.database_config import DATABASE_CLIENT, DATABASE_NAME, DATABASE_COLLECTION

# Connect to the mongodb client
client = MongoClient(DATABASE_CLIENT)
db = client[DATABASE_NAME]
collection = db[DATABASE_COLLECTION]