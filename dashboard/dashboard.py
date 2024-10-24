import typing as T

import cv2
import numpy as np
import streamlit as st
from dataset.dataset import CoralDataset

st.set_page_config(layout="wide")
st.title("Coral Dashboard")


class CoralDashboard:
    def __init__(self):
        self.dataset = CoralDataset()
        self.num_cols = 18
        self.col_idx = -1
        self.num_samples = -1
        self.setting_col_idx = -1
        self.settings()

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

    def display_sample(self, sample) -> None:
        mask = sample.mask.permute(1, 2, 0).numpy().astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.squeeze() * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image_np = sample.image.permute(1, 2, 0).numpy().copy()
        cv2.drawContours(image_np, contours, -1, (0, 255, 0), 2)
        st.image(image_np)

    def run(self) -> None:
        cols = st.columns(self.num_cols)
        for i in range(self.num_samples):
            sample = self.dataset[i]
            with cols[self.next_col_idx()]:
                self.display_sample(sample)


if __name__ == "__main__":
    dashboard = CoralDashboard()
    dashboard.run()
