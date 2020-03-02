# Know Your Enemies and Know Yourself in the Real-Time Bidding Function Optimisation

The code for supporting the paper under review.

## Datasets
* `iPinYou`. 
We removed the feature `IP` and `advertiser` in the feature list in file `make-ipinyou-data/python/mkyzx.py`
and followed the same data-preprocess procedure on [this page](https://github.com/wnzhang/make-ipinyou-data).

### Prepare the dataset
* Prepare `iPinYou` dataset as described [here](https://github.com/wnzhang/make-ipinyou-data) and put `make-ipinyou-data` folder in the same parent folder as `anonymous-submission` project.
* Copy the `info.txt` file from the `info/2259/` folder to `data/make-ipinyou-data/2259/`

```
|-- anonymous-submission
----|-- data
--------|-- make_ctr_ipinyou_dataset.py
--------|-- make-ipinyou-data
------------|-- 2259
------------...
----|-- example
----|-- marl_bidding
----|-- market_modelling
----|-- master_config.py
```

### Train the CTR model
Run the `make_ctr_ipinyou_dataset.py` script in the `data` folder to generate data files with CTR values for each bid log which is stored as: `train.ctr.txt`
```
python make_ctr_ipinyou_dataset.py
```

## Run the experiments
### Single DDPG agent vs. Linear agents
```
./run_single_ddpg_vs_linear.sh
```

### Multiple DDPG agents 
```
./run_multi_ddpg.sh
```






