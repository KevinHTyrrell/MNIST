import json


def load_config(filename: str):
    with open(filename, 'r') as f:
        raw_config_str = f.read()
    if filename.find('.json') != -1:
        return json.loads(raw_config_str)
    return raw_config_str


def convert_json_list_tuple(json_data: dict):
    for k, v in json_data.items():
        if isinstance(json_data.get(k), list):
            json_data.update({k: tuple(v)})
    return json_data