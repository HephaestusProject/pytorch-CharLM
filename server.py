from pydantic import BaseModel

from serving.app_factory import create_app


class Request(BaseModel):
    input_text: str


class Response(BaseModel):
    generated_text: str


def handler(request: Request) -> Response:
    return "hi"


app = create_app(handler, Request, Response)
