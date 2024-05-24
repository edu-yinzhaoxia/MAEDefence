import numpy as np
import random
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def predict(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
        pred = torch.argmax(outputs, dim=1)
        return pred

def attack_success(targets, pred):
    attack_id = np.where(targets.cpu() == pred.cpu())[0]
    return attack_id





