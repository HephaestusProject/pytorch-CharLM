from pathlib import Path

from pydantic import BaseModel

from predictor import Predictor
from serving.app_factory import create_app


predictor = Predictor.from_checkpoint(
    checkpoint_path=Path("checkpoints/epoch=024_val_ppl=101.52542.ckpt"),
)


class Request(BaseModel):
    input_text: str


class Response(BaseModel):
    generated_text: str


def handler(request: Request) -> Response:
    prediction = predictor.predict(input_text=request.input_text)
    return Response(generated_text=prediction)


app = create_app(handler, Request, Response)
