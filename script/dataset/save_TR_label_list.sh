#!/bin/bash
set -e

# bash script/dataset/save_TR_label_list.sh

python -m app.dataset.save_TR_label_list --purpose train --n_node 20 --batch_size 10

python -m app.dataset.save_TR_label_list --purpose val --n_node 20 --batch_size 10

python -m app.dataset.save_TR_label_list --purpose test --n_node 20 --batch_size 10
python -m app.dataset.save_TR_label_list --purpose test --n_node 50 --batch_size 20
python -m app.dataset.save_TR_label_list --purpose test --n_node 100 --batch_size 30
python -m app.dataset.save_TR_label_list --purpose test --n_node 500 --batch_size 100
python -m app.dataset.save_TR_label_list --purpose test --n_node 1000 --batch_size 200