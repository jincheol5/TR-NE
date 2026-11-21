#!/bin/bash
set -e

python -m app.experiment.train_model --model_name TGAT --seed 1 --lr 0.0001

python -m app.experiment.train_model --model_name TGN --seed 1 --lr 0.0001

python -m app.experiment.train_model --model_name TR-NE --seed 1 --lr 0.0001