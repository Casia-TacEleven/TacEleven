import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import uvicorn
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
import argparse

from models.TacGen_once_predict_variable import TacGen
from models.train_dist_sr import LanguageEmbedding
from utils.utils_ import load_weights, WholeDataset, collate_fn_self_regression

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the FastAPI server with configurable parameters.")
parser.add_argument("--temperature", type=float, default=1, help="Temperature for inference.")
parser.add_argument("--port", type=int, default=8099, help="Port to run the FastAPI server.")
parser.add_argument("--device", type=int, default=0, help="Device ID to run the model on.")

args = parser.parse_args()

TEMPERATURE = args.temperature
PORT = args.port
DEVICE = args.device

@dataclass
class InferenceRequest:
    X: list  # input history data with shape(batch, time=32, node=23, dim=2)
    TE_x: list  # input history data with shape(batch, time=32, node=23, dim=2)
    TE_y: list
    lang: str  # Language description with shape(batch, word_count)

# Initialize model
model = TacGen(
    L=6,
    K=4,
    d=256
)
print('Initialized model parameters: {:,}'.format(model.count_parameters()))

app = FastAPI()
device = f'cuda:{DEVICE}'
model.use_cuda(device)
# Load trained model weights
load_weights(
    model=model,
    weight_path='/home/trl/fllm/gman_lc/saves/full/20250804_221642_pmcc8h1jSR8K6TT8/ckpt.pkl',
    device=device
)

@app.post("/predict/")
def predict(request: InferenceRequest):
    # try:
        # Parse input data
    print(request.X)
    X = torch.tensor(request.X).float().to(device)
    TE_x = torch.tensor(request.TE_x).float().to(device)
    TE_y = torch.tensor(request.TE_y).float().to(device)
    print(X.shape)
    lang_list = [request.lang]
    if type(lang_list) is str:
        lang_list = [lang_list]
    assert type(lang_list) is list

    # Perform inference
    with torch.no_grad():
        out = model(
            X=X,
            TE_x=TE_x,
            TE_y=TE_y,
            lang_list=lang_list,
            temperature=TEMPERATURE
        )
        if isinstance(out, tuple):
            out_pos, mu, logvar = out
        else:
            out_pos = out
    print(f'out_pos is {out_pos[0,0,0,:]}')

    return {
        "prediction": out_pos.cpu().numpy().tolist(),
    }
    # except Exception as e:
    #     print(e)
    #     raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
