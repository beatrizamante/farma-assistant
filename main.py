from _lib.container import get_container
from config.logger import setup_logging
from fastapi import FastAPI

from src.interface.http.server import make_server
from src.interface.http.routes.prompt_route import router as prompt_route

logger = setup_logging()
container = get_container()

app = FastAPI()

app.include_router(prompt_route)

if __name__ == "__main__":
    logger.info("Starting WatchMe AI Backend...")
    server = make_server()
    server.run()
