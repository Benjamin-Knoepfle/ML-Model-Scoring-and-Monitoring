import json

def load_configuration(mode:str):
    # Load config.json and get input and output paths
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config[mode]