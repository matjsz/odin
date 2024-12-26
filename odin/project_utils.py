from gzip import READ
import os
import yaml


def get_project_info():
    path = os.path.abspath(".") + "\\project.yaml"

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return data


def get_project_pretty_type(type):
    types = {"detection": "Object Detection", "classification": "Image Classification"}
    return types[type]
