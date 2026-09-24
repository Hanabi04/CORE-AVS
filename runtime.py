import random
import numpy as np
import torch
from models import SetAuditor


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(checkpoint, data, device="cpu", batch_size=128):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = SetAuditor().to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    mean = ck["norm"]["mean"].cpu().numpy()
    scale = ck["norm"]["scale"].cpu().numpy()
    x = torch.from_numpy((data["features"] - mean) / scale)
    risks, probabilities = [], []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            risk, logits = model(x[start : start + batch_size].to(device))
            risks.append(risk.cpu().numpy())
            probabilities.append(logits.softmax(-1).cpu().numpy())
    return {
        "risk": np.concatenate(risks),
        "condition_probability": np.concatenate(probabilities),
    }, ck
