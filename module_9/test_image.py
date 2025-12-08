import onnxruntime as ort
from PIL import Image
from urllib import request
import numpy as np

# Load the correct ONNX model file
ort_session = ort.InferenceSession("/workdir/hair_classifier_v1.onnx")
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Download image
url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
with request.urlopen(url) as resp:
    img = Image.open(resp)

# Preprocess
img = img.convert('RGB').resize((200, 200), Image.NEAREST)
img_np = np.array(img).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
std  = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
img_np = (img_np - mean) / std
input_tensor = np.transpose(img_np, (2,0,1))[None, :].astype(np.float32)

# Run inference
pred = ort_session.run([output_name], {input_name: input_tensor})
print("Model output:", float(pred[0][0][0]))
