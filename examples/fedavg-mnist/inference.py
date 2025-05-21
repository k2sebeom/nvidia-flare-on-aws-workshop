import torch
import json


def model_fn(model_dir):
    model = torch.jit.load(f"{model_dir}/model.pt")
    model.eval()
    return model

def predict_fn(input_data, model):
    with torch.no_grad():
        return model(input_data)

def input_fn(request_body, request_content_type):
    # Convert input data to tensor
    assert request_content_type == 'application/json'
    data = json.loads(request_body)['inputs']
    data = torch.tensor(data, dtype=torch.float32)
    return data

def output_fn(prediction, content_type):
    # Convert prediction to JSON
    assert content_type == 'application/json'
    res = prediction.cpu().numpy().tolist()
    return json.dumps(res)
