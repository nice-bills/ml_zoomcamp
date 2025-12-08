from io import BytesIO
from urllib import request
from PIL import Image
import torch
from torchvision import transforms
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("hair_classifier_v1.onnx")
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    img = Image.open(BytesIO(buffer))
    return img

def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict_image(url):
    img = download_image(url)
    img = prepare_image(img)
    
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_np).permute(2, 0, 1)
    img_tensor = preprocess(img_tensor)
    input_tensor = img_tensor.unsqueeze(0).numpy().astype(np.float32)
    
    pred = ort_session.run([output_name], {input_name: input_tensor})
    return float(pred[0][0][0])
