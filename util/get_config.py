import argparse
from dataclasses import fields
from pathlib import Path

from config.config import Config

OUTPUT_DIR = Path("/mnt/windows/coral_data/logs")


def get_config() -> Config:
    parser = argparse.ArgumentParser(description="Train the Coral Network")
    config = Config()
    for field in fields(config):
        kwargs = {"help": field.metadata.get("help", "")}
        if field.type is bool:
            print("here")
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = field.type
            kwargs["default"] = None
        parser.add_argument(f"--{field.name}", **kwargs)
    args = parser.parse_args()
    for field in fields(config):
        arg = getattr(args, field.name)
        if arg is not None:
            print(f"Overriding {field.name} -> {arg}")
            setattr(config, field.name, arg)
    if not config.output_dir:
        config.output_dir = OUTPUT_DIR
    return config
