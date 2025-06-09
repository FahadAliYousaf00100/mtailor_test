from model import LoadModel, Prediction
from PIL import Image
from fastapi import FastAPI


model = None


def load_model(app: FastAPI):
    """
    Load the model at startup.
    """
    global model
    model = LoadModel()
    yield


app = FastAPI(lifespan=load_model)


@app.get("/predict")
def predict(image_path: str):
    img = Image.open(image_path)
    prediction = Prediction(model)
    prediction.pre_process(img)
    predicted_class = prediction.predict()
    return {"predicted_class": int(predicted_class[0])}
