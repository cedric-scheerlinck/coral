import typing as T
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from config.config import Config
from dataset.dataset import CoralDataset
from dataset.dataset import Sample
from model.model import CoralModel
from util.image_util import numpy_from_torch

st.set_page_config(layout="wide")
st.title("Coral Dashboard")


def get_checkpoint_path(model_path: str) -> T.Optional[Path]:
    if not model_path:
        return None
    path = Path(model_path)
    if path.is_file():
        return path
    if not path.exists():
        st.error(f"Path {path} does not exist")
        return None
    checkpoints = sorted(path.glob("**/*.ckpt"))
    if not checkpoints:
        st.error(f"No checkpoint found in {path}")
        return None
    return checkpoints[-1]


class CoralDashboard:
    def __init__(self):
        self.config = Config()
        self.dataset = CoralDataset(self.config)
        self.num_cols = 18
        self.num_samples = -1
        self.model_path = ""
        self.col_idx = -1
        self.setting_col_idx = -1
        self.model: T.Optional[CoralModel] = None

    def run(self) -> None:
        self.settings()
        self.model = self.create_model()
        self.display_samples()

    def create_model(self) -> T.Optional[CoralModel]:
        checkpoint_path = get_checkpoint_path(self.model_path)
        if not checkpoint_path:
            return None
        model = CoralModel.load_from_checkpoint(checkpoint_path, config=self.config)
        model.eval()
        return model

    def display_samples(self) -> None:
        cols = st.columns(self.num_cols)
        for i in range(self.num_samples):
            sample_dict = self.dataset[i]
            sample = Sample.from_dict(sample_dict)
            with cols[self.next_col_idx()]:
                self.display_sample(sample)
                if self.model:
                    self.display_output(sample)

    def display_output(self, sample: Sample) -> None:
        pred = self.model.get_pred(sample.image)
        sample = Sample(image=sample.image, mask=pred)
        self.display_sample(sample)

    def settings(self) -> None:
        cols = st.columns(6)
        with cols[self.next_setting_col_idx()]:
            self.create_param(st.number_input, "num_cols", min_value=1)
        with cols[self.next_setting_col_idx()]:
            self.create_param(
                st.number_input,
                "num_samples",
                min_value=-1,
                max_value=len(self.dataset),
            )
            if self.num_samples == -1:
                self.num_samples = len(self.dataset)
        with cols[self.next_setting_col_idx()]:
            self.create_param(st.text_input, "model_path")

    def next_setting_col_idx(self) -> int:
        self.setting_col_idx = (self.setting_col_idx + 1) % self.num_cols
        return self.setting_col_idx

    def next_col_idx(self) -> int:
        self.col_idx = (self.col_idx + 1) % self.num_cols
        return self.col_idx

    def create_param(
        self, widget: st.delta_generator, name: str, **kwargs: T.Any
    ) -> None:
        init_value = getattr(self, name)
        if name in st.query_params:
            init_value = type(init_value)(st.query_params[name])

        param = widget(label=name, value=init_value, **kwargs)
        setattr(self, name, param)
        st.query_params[name] = param

    def display_sample(self, sample: Sample) -> None:
        mask = numpy_from_torch(sample.mask)
        contours, _ = cv2.findContours(
            (mask.squeeze() > 127).astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        image_np = numpy_from_torch(sample.image)
        cv2.drawContours(image_np, contours, -1, (0, 255, 0), 2)
        st.image(image_np)
        st.image(mask)


if __name__ == "__main__":
    dashboard = CoralDashboard()
    dashboard.run()
