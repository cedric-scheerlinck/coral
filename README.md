# Coral Segmentation Model

Simple deep learning coral polyp image segmentation project. Create a dataset, train a model, run inference and dashboard.

## Quick Start

1. Clone this repo
```
git clone https://github.com/cedric-scheerlinck/coral.git
cd coral
```
2. Set up a python [virtual environment](https://docs.python.org/3/library/venv.html)
```
python3 -m venv venv
source venv/bin/activate
export PYTHONPATH=$(pwd)
```
3. Install requirements
```
pip install -r requirements.txt
```
4. Grab a model: https://drive.google.com/drive/folders/1l39sCAX4WQAKrxkjDPPlzatjfcVS2gVI?usp=sharing
5. Run inference on an image
```
python executable/inference.py <path/to/model> <path/to/image>
```
