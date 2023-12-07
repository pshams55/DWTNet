import os
import importlib
from .base_model import BaseModel


def find_model_using_name(model_name):
    
    model_file_name = "model." + model_name + "_model"
    modellib = importlib.import_module(model_file_name)
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == model_name.lower() and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_file_name, model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    
    model = find_model_using_name(model_name)
    return model.modify_options


def create_model(opt):
    
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
