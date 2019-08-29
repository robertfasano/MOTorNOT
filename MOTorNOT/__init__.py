import ruamel_yaml as yaml
import os

def load_parameters():
    path = os.path.join(os.path.dirname(__file__), 'parameters.yml')
    with open(path) as file:
        params = yaml.load(file, Loader=yaml.SafeLoader)
    return params
