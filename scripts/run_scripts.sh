#!/bin/bash

python lightning_train.py "configs/md17/Fractal_EGNN_v2_S.yaml"
python lightning_train.py "configs/md17/EGNN.yaml"
python lightning_train.py "configs/md17/Fractal_EGNN_v2_L.yaml"
