import os.path

from src.yolov6.utils.config import Config

def create_model(model_name):
    cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), f'configs/{model_name}.py'))
    model = cfg.model.type(cfg.model)
    return model, cfg
