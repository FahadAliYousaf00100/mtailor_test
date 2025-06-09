from convert_to_onnx import convert_to_onnx, create_torch_model
import torch
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image


class LoadModel:
    model_loades = False

    def __new__(cls):
        if not cls.model_loades:
            cls.model_loades = True
            state = "./pytorch_model_weights.pth"
            output_path = "./pytorch_model.onnx"
            model = create_torch_model(state)
            input_tensor = torch.randn(1, 3, 224, 224)
            convert_to_onnx(model, input_tensor, output_path)
            cls.model = ort.InferenceSession(output_path)

        return cls.model


class Prediction:
    def __init__(self, model):
        self.model = model
        self.pre_processed_image = None

    def predict(self):
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: self.pre_processed_image})
        output = outputs[0]
        predicted_class = np.argmax(output, axis=1)
        return predicted_class

    def pre_process(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize((224, 224), Image.BILINEAR)
        img = np.asarray(img).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        self.pre_processed_image = img
