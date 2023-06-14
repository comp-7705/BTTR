from typing_extensions import Annotated
import base64

import numpy as np
import cv2 as cv
from torchvision.transforms import ToTensor
from PIL import Image

from bttr.lit_bttr import LitBTTR

from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()
device = "cpu"


def init_model():
    ckp_path = "pretrained-2014.ckpt"
    model = LitBTTR.load_from_checkpoint(ckp_path, map_location=device)
    model.eval()
    # print("Init model")

    return model


class Buffer(BaseModel):
    payload: str


@app.post("/api/bttr/prediction/")
async def predict(buffer: Buffer, model: Annotated[LitBTTR, Depends(init_model)]):
    img_buffer = base64.b64decode(buffer.payload)
    img_array = np.frombuffer(img_buffer, np.uint8)
    img = cv.imdecode(img_array, cv.IMREAD_ANYDEPTH)
    img = Image.fromarray(np.array(img))
    # im.show()

    img = ToTensor()(img).to(device)
    pred_latex = model.beam_search(img)
    # print("Prediciton:", pred_latex)

    return pred_latex
