import typing as T
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from config.config import Config
from dataset.dataset import CoralDataset, Sample
from model.model import CoralModel
from util.constants import Split
from util.image_util import numpy_from_torch

st.set_page_config(layout="wide")
st.title("Coral Dashboard")


def get_checkpoint_path(model_path: str) -> Path | None:
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
        self.data_path = ""
        self.num_cols = 5
        self.num_samples = -1
        self.num_samples_per_tile = 5
        self.line_thickness = 2
        self.model_path = ""
        self.col_idx = -1
        self.setting_col_idx = -1
        self.start_idx = 0
        self.stride = 1
        self.split: Split = "train"
        self.model: CoralModel | None = None

    def run(self) -> None:
        self.settings()
        self.model = self.create_model()
        self.display_samples()

    def create_model(self) -> CoralModel | None:
        checkpoint_path = get_checkpoint_path(self.model_path)
        if not checkpoint_path:
            return None
        model = CoralModel.load_from_checkpoint(checkpoint_path, config=self.config)
        model.eval()
        model = model.cpu()
        return model

    def get_sample_indices(self) -> T.Any:
        stop_idx = self.num_samples * self.stride
        for i in range(self.start_idx, stop_idx, self.stride):
            if i >= len(self.dataset):
                return
            yield i

    def display_samples(self) -> None:
        cols = st.columns(self.num_cols)
        seen = Counter()
        st.text(len(self.dataset.sample_paths))
        for sample_path in self.dataset.sample_paths:
            name = sample_path.parent.name
            if seen[name] >= self.num_samples_per_tile:
                continue
            sample_dict = self.dataset.load_path(sample_path)
            sample = Sample.from_dict(sample_dict)
            if not sample.mask.any():
                continue
            seen[name] += 1
            with cols[self.next_col_idx()]:
                self.display_sample(sample)
                if self.model:
                    self.display_output(sample)

    def display_output(self, sample: Sample) -> None:
        pred = self.model.get_pred(sample.image)
        sample = Sample(image=sample.image, mask=pred, path=sample.path)
        sample = replace(sample, mask=pred)
        self.display_sample(sample)

    def settings(self) -> None:
        cols = st.columns(6)
        with cols[self.next_setting_col_idx()]:
            self.create_param(st.text_input, "data_path")
            if self.data_path:
                self.config.data_dir = self.data_path
        with cols[self.next_setting_col_idx()]:
            options: T.List[Split] = ["train", "val", "test"]
            self.create_param(st.selectbox, "split", options=options)
        self.config.split = self.split
        self.dataset = CoralDataset(self.config)
        with cols[self.next_setting_col_idx()]:
            self.create_param(st.number_input, "start_idx", min_value=0)
        with cols[self.next_setting_col_idx()]:
            self.create_param(st.number_input, "stride", min_value=1)
        with cols[self.next_setting_col_idx()]:
            self.create_param(st.number_input, "num_cols", min_value=1)
        with cols[self.next_setting_col_idx()]:
            self.create_param(st.number_input, "num_samples_per_tile", min_value=1)
        with cols[self.next_setting_col_idx()]:
            self.create_param(st.number_input, "line_thickness", min_value=1)
        with cols[self.next_setting_col_idx()]:
            self.create_param(
                st.number_input,
                "num_samples",
                min_value=-1,
                max_value=len(self.dataset),
            )
            if self.num_samples == -1:
                self.num_samples = len(self.dataset)
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

        if "options" in kwargs:
            kwargs["index"] = kwargs["options"].index(init_value)
        else:
            kwargs["value"] = init_value

        param = widget(label=name, **kwargs)
        setattr(self, name, param)
        st.query_params[name] = param

    def display_sample(self, sample: Sample) -> None:
        mask = numpy_from_torch(sample.mask)
        kernel_size = 41
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        cv2.circle(
            kernel, (kernel_size // 2, kernel_size // 2), kernel_size // 2, 1, -1
        )
        mask = cv2.dilate(mask, kernel)
        contours, _ = cv2.findContours(
            (mask.squeeze() > 127).astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        image_np = numpy_from_torch(sample.image)
        cv2.drawContours(
            image_np, contours, -1, (0, 255, 0), thickness=self.line_thickness
        )
        space = " " * 100 + "."
        st.text(
            f"{Path(sample.path).parents[1].name} | {Path(sample.path).parent.name}"
            + space,
            help=f"""```
            {sample.path}""",
        )
        st.image(image_np)


if __name__ == "__main__":
    dashboard = CoralDashboard()
    dashboard.run()
