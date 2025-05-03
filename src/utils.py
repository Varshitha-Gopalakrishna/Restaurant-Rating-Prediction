import dill
import os

def load_model(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
    with open(path, "rb") as f:
        return dill.load(f)