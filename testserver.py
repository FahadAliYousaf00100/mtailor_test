from model import LoadModel, Prediction
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from io import BytesIO

model = None


def load_model(app: FastAPI):
    """
    Load the model at startup.
    """
    global model
    model = LoadModel()
    yield


app = FastAPI(lifespan=load_model)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    # img = Image.open(image_path)
    prediction = Prediction(model)
    prediction.pre_process(img)
    predicted_class = prediction.predict()
    return {"predicted_class": int(predicted_class[0])}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/ready")
def readiness_check():
    global model
    if model is not None:
        return {"status": "ready"}
    else:
        return {"status": "not ready"}, 503
