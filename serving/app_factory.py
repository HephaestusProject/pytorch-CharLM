from fastapi import FastAPI
from typing import Any, Callable

def create_app(handler: Callable, request_type: Any, response_type: Any):

    app = FastAPI()
    
    @app.post("/model/")
    async def inference(request: request_type) -> response_type:
        response = handler(request)
        return response

    return app