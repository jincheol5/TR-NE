#!/bin/bash
set -e

# bash script/dataset/save_graph_list.sh

python -m app.dataset.save_graph_list --n_node 20 --n_graph 100 --purpose train

python -m app.dataset.save_graph_list --n_node 20 --n_graph 5 --purpose val

python -m app.dataset.save_graph_list --n_node 20 --n_graph 5 --purpose test
python -m app.dataset.save_graph_list --n_node 50 --n_graph 5 --purpose test
python -m app.dataset.save_graph_list --n_node 100 --n_graph 5 --purpose test
python -m app.dataset.save_graph_list --n_node 500 --n_graph 5 --purpose test
python -m app.dataset.save_graph_list --n_node 1000 --n_graph 5 --purpose test