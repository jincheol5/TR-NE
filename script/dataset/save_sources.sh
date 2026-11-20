#!/bin/bash
set -e

# bash script/dataset/save_sources.sh

python -m app.dataset.save_sources --n_node 20 --purpose val 

python -m app.dataset.save_sources --n_node 20 --purpose test 

python -m app.dataset.save_sources --n_node 50 --purpose test 

python -m app.dataset.save_sources --n_node 100 --purpose test 

python -m app.dataset.save_sources --n_node 500 --purpose test 

python -m app.dataset.save_sources --n_node 1000 --purpose test 
