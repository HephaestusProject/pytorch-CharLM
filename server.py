from pydantic import BaseModel

from serving.app_factory import create_app


class Request(BaseModel):
    base64_image_string: str


class Response(BaseModel):
    prediction: str


def handler(request):
    return "hi"


app = create_app(handler, Request, Response)
