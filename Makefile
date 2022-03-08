.PHONY: all setup datasets test* clear_experiments

all : setup datasets

setup:
	python3 -m pip install -r requirements.txt

datasets:
	mkdir -p datasets/
	wget -O datasets/btc_h.csv "http://static.trality.com/fire/data/btc_h.csv"

test_multi:
	python main.py parameters/sin_wave_test_multi.json

test_single:
	python main.py parameters/autoregressive_test_single.json

clear_experiments:
	sh scripts/clear_experiments.sh
