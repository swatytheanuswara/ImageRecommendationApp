from fastapi import FastAPI
from apps.routes import view
from config import cors

app = FastAPI(title="AI IMAGE SEARCH AND RECOMMENDATION SYSTEM", middleware=cors.middleware)

# Load API's
app.include_router(view.router)

