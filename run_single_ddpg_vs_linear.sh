#!/bin/bash

agent=3
noise=0.001
reward=2
q='indi'
op='tf'

for camp in 2259;
do
# for value in 0.125 0.25 0.5;
for value in 0.125;
do
# for seed in 0 15 28;
for seed in 0;
do
    echo "Run the baseline experiment. DDPG agent without opponent model vs. linear bidders"
    cd example/
    python ddpg_linear_with_test.py --om=False --budget=$value --camp=$camp --agent-num=$agent --noise=$noise --reward=$reward --seed=$seed --lin-ratio=100

    # Train DASA model as the opponent model
#    cd ../market_modelling/
#    echo "start training the market $value $seed"
#    python market_predictor.py --train=True --budget=$value --camp=$camp --multi=False --agent-index=None --agent-num=$agent --noise=$noise --reward=$reward --seed=$seed --q-func=$q --op=$op
#    python market_predictor.py --train=False --budget=$value --camp=$camp --multi=False --test-file='train.ctr.txt' --agent-index=None --agent-num=$agent --noise=$noise --reward=$reward --seed=$seed --q-func=$q --op=$op
#
#
#    echo "train ddpg with om $value $seed"
#    cd ../example/
#    python ddpg_linear_with_test.py --om=True --budget=$value --camp=$camp --agent-num=$agent --noise=$noise --reward=$reward --seed=$seed --lin-ratio=100
#    python ddpg_linear_with_test.py --om=True --budget=$value --train-mode=False --camp=$camp --agent-num=$agent --noise=$noise --reward=$reward --seed=$seed --lin-ratio=100
#
#
#    echo "read results $value $seed"
#    python read_single_results.py --budget=$value --camp=$camp --noise=$noise --reward=$reward --seed=$seed --agent-num=$agent

    cd ../

done;
done;
done
