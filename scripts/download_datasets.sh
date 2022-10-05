mkdir -p datasets/

# Stocks datasets
python scripts/yfinance_downloader.py --ticker AAPL  --target datasets/aapl_d.csv
python scripts/yfinance_downloader.py --ticker SPY  --target datasets/spy_d.csv

# Crypto datasets
python scripts/binance_downloader.py --interval 1h --symbol BTCUSDT --target datasets/btc_h.csv
python scripts/binance_downloader.py --interval 1h --symbol ETHUSDT --target datasets/eth_h.csv
python scripts/binance_downloader.py --interval 1h --symbol XRPUSDT --target datasets/xrp_h.csv
