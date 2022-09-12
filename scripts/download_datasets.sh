mkdir -p datasets/

# Stocks datasets
python scripts/yfinance_downloader.py --ticker TSLA  --target datasets/tsla_d.csv

# Crypto datasets
python scripts/binance_downloader.py --interval 1h --symbol BTCUSDT --target datasets/btc_h.csv
python scripts/binance_downloader.py --interval 1h --symbol ETHUSDT --target datasets/eth_h.csv
