import argparse
from dataclasses import fields
from pathlib import Path

from config.config import Config

OUTPUT_DIR = Path("/media/cedric/Storage1/coral_data/logs")


def get_config() -> Config:
    parser = argparse.ArgumentParser(description="Train the Coral Network")
    config = Config()
    for field in fields(config):
        parser.add_argument(
            f"--{field.name}",
            type=field.type,
            default=None,
            help=field.metadata.get("help", ""),
        )
    args = parser.parse_args()
    for field in fields(config):
        arg = getattr(args, field.name)
        if arg is not None:
            print(f"Overriding {field.name} -> {arg}")
            setattr(config, field.name, arg)
    if not config.output_dir:
        config.output_dir = OUTPUT_DIR
    return config
