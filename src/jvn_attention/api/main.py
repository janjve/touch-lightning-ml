from fastapi import FastAPI
from jvn_attention.xor import inference

app = FastAPI()

xor_predictor = inference.XORPredictor()


@app.post("/xor/")
async def index(x: inference.InputRow) -> inference.Prediction:
    return xor_predictor(x)
