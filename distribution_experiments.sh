#!/bin/bash
CRYPTO_DATASETS="datasets/btc_h.csv  datasets/eth_h.csv  datasets/xrp_h.csv"
STOCK_DATASETS="datasets/aapl_d.csv datasets/spy_d.csv"

for DATASET in $CRYPTO_DATASETS
do
    python multiple_run.py parameters/new_plots/template_crypto.json $DATASET &
done

for DATASET in $STOCK_DATASETS
do
    python multiple_run.py parameters/new_plots/template_stock.json $DATASET &
done