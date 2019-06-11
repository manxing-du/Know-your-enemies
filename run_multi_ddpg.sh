#!/bin/bash


agent=3
noise=0.001
reward=2
q='indi'
op='tf'

for camp in 2259;
do
for value in 0.125 0.25 0.5;
do
for seed in 0 15 28;
do

    echo "start processing budget c0=" $value

    echo "run ddpg training without om and log the price"
    cd example/
    python multi_ddpg_agents.py --om=False --budget=$value --camp=$camp --reward=$reward --seed=$seed --agent-num=$agent --q-func=$q --noise=$noise

    echo "start training DASA model for every agent"
    cd ../market_modelling/
    # agent-index : range(N), N=agent_numer
    for label in 0 1 2;
    do
        python market_predictor.py --train=True --budget=$value --camp=$camp --agent-index=$label --agent-num=$agent --reward=$reward --seed=$seed --noise=$noise --q-func=$q --op=$op
        python market_predictor.py --train=False --budget=$value --camp=$camp --test-file='train.ctr.txt' --agent-index=$label --agent-num=$agent --reward=$reward --seed=$seed --noise=$noise --q-func=$q --op=$op
        python market_predictor.py --train=False --budget=$value --camp=$camp --test-file='test.ctr.txt' --agent-index=$label --agent-num=$agent --reward=$reward --seed=$seed --noise=$noise --q-func=$q --op=$op
    done;

    # run ddpg with opponent model
    cd ../example/
    python multi_ddpg_agents.py --om=True --budget=$value --camp=$camp --random-om=False --reward=2 --seed=$seed --agent-num=$agent --noise=$noise --q-func=$q
    cd ../

done;
done;
done