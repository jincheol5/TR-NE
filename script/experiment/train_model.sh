#!/bin/bash

# MODELS=("TGAT" "TGN" "TR-NE")
MODELS=("GraphMixer")
SEEDS=(1 2 3)
LRS=(0.0001 0.0005)

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for LR in "${LRS[@]}"; do
            echo "Running: model=$MODEL, seed=$SEED, lr=$LR"

            python -m app.experiment.train_model \
                --model_name "$MODEL" \
                --seed "$SEED" \
                --lr "$LR" \
                --use_mlflow 1 \
                --experiment_name TR-NE \
                --save_model 1
        done
    done
done