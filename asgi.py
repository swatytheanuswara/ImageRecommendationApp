import os
import uvicorn
from config import env_config
from apps.__init__ import app


if __name__ == "__main__":
    uvicorn.run("asgi:app",
                host=os.environ.get('SERVER_HOST'),
                port=int(os.environ.get('SERVER_PORT')),
                reload=True)