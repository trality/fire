import argparse
import yfinance as yf

def run(ticker, target):
    df = yf.download(ticker, start="2002-02-01", end="2022-09-01")
    del df['Close']
    df = df.rename({'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Adj Close': 'close',
                    'Volume': 'volume'}, axis='columns')
    df.index.rename('start', inplace=True)
    df.to_csv(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", help="ticker to download", type=str)
    parser.add_argument("--target", help="Target file", type=str)
    run(**vars(parser.parse_args()))
