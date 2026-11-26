#!/bin/bash

# MODELS=("TGAT" "TGN" "TR-NE")
MODELS=("GraphMixer")
SEEDS=(1 2 3)
LRS=(0.0001 0.0005)

N_NODES=(20 50 100 500 1000)
BATCH_SIZES=(10 20 30 100 200)

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for LR in "${LRS[@]}"; do
            for i in "${!N_NODES[@]}"; do
                N_NODE=${N_NODES[$i]}
                BATCH_SIZE=${BATCH_SIZES[$i]}

                echo "Testing: model=$MODEL, seed=$SEED, lr=$LR, n_node=$N_NODE, batch_size=$BATCH_SIZE"

                python -m app.experiment.test_model \
                    --model_name "$MODEL" \
                    --seed "$SEED" \
                    --lr "$LR" \
                    --n_node "$N_NODE" \
                    --batch_size "$BATCH_SIZE" \
                    --experiment_name TR-NE \
                    --save_metric 1
            done
        done
    done
done