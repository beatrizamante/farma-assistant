from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.interface.http.server import make_server
from src.interface.http.routes.prompt_route import router as prompt_route
from src._lib.container import get_container

CONTAINER = get_container()
logger = CONTAINER.logger()

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initializes modules on app start instead of waiting for a request"""
    logger.info("Loading model and tokenizer...")
    CONTAINER.model()
    CONTAINER.tokenizer()
    logger.info("Model and tokenizer ready.")
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(prompt_route)

if __name__ == "__main__":
    logger.info("Starting WatchMe AI Backend...")
    server = make_server()
    server.run()
