from serving.app_factory import create_app

from pydantic import BaseModel


class Request(BaseModel):
    base64_image_string: str


class Response(BaseModel):
    prediction: str

def handler(request):
    return "hi"

app = create_app(handler, Request, Response)
