from predictor import Predictor
from pathlib import Path


def test_predictor():
    predictor = Predictor.from_checkpoint(
        checkpoint_path=Path("checkpoints/epoch=024_val_ppl=101.52542.ckpt"),
    )

    prediction = predictor.predict(input_text="if")
    print(prediction)
