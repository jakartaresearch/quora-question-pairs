import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from pydantic import BaseModel
from app.model import ParaphraseIdentifier_Model
from app.engine import ParaphraseIdentifier_Engine


app = FastAPI()

model = ParaphraseIdentifier_Model('model')
engine = ParaphraseIdentifier_Engine(model)


class Item(BaseModel):
    q1: str
    q2: str


@app.get("/")
def read_root():
    return {'message': 'API is running...'}


@app.post("/paraphrase/predict")
def predict(item:Item):
    item = item.dict()
    engine.predict(item['q1'], item['q2'])
    return engine.output
