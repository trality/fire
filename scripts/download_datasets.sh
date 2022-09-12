mkdir -p datasets/
python scripts/binance_downloader.py --interval 1h --symbol BTCUSDT --target datasets/btc_h.csv
python scripts/binance_downloader.py --interval 1h --symbol ETHUSDT --target datasets/eth_h.csv
