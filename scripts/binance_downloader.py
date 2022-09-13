import argparse
import requests
import time
import pandas as pd

ENDPOINT = 'https://api.binance.com'
CANDLE_LIMIT = 1000
SLEEP_TIME = 0.2


def current_milli_time():
    return round(time.time() * 1000)


def fetch_klines(symbol, interval, startTime, endTime):
    url = ENDPOINT + '/api/v3/klines'
    r = requests.get(url=url, params={'symbol': symbol, 'interval': interval,
                     'startTime': startTime, 'endTime': endTime, 'limit': CANDLE_LIMIT})
    return r.json()


def check_api_call(symbol, interval, lower, upper):
    d = fetch_klines(symbol, interval, lower, upper)
    if ('msg' in d):
        raise RuntimeError(d['msg'])


def fetch_candle_list(symbol, interval, lower, upper):
    kline_list = []
    current_lower = lower
    progress = 0

    check_api_call(symbol, interval, lower, upper)

    while progress < 1:
        kline_list.extend(fetch_klines(symbol, interval, current_lower, upper))
        current_lower = kline_list[-1][6]
        progress = (current_lower-kline_list[0][0]) / (current_milli_time()-kline_list[0][0])
        print(f' fetching {symbol} candles: {min(1,progress):.2%}', end="\r")
        time.sleep(SLEEP_TIME)
    print('')
    return kline_list


def fetch_dataset(lower, interval, symbol):
    upper = current_milli_time()

    kline_list = fetch_candle_list(symbol, interval, lower, upper)
    df = pd.DataFrame(kline_list)
    df.columns = ['start', 'open', 'high', 'low', 'close', 'volume', 'end',
                  'quote_asset_volume', 'trades', 'buy_base_volume', 'buy_quote_volume', 'ignore']

    df = df[['start', 'end', 'open', 'high', 'low', 'close', 'volume']]
    df['start'] = pd.to_datetime(df['start'], unit='ms')
    df['end'] = pd.to_datetime(df['end'], unit='ms')

    float_columns = ['open', 'high', 'low', 'close', 'volume']
    for c in float_columns:
        df[c] = df[c].astype(float)

    df.set_index('start', inplace=True)

    return df


def remove_duplicates(df, verbose=True):
    duplicated = df[df.index.duplicated(keep=False)]
    if verbose and len(duplicated) > 0:
        print(f"There are {len(duplicated)} duplicated rows")
    return df[~df.index.duplicated(keep='first')]


def clean_df(df: pd.DataFrame):
    df = df.dropna()
    df.index = df.index.astype('datetime64[ns]')
    df = df.sort_index()
    df = remove_duplicates(df)


def run(lower_bound, interval, symbol, target):
    df = fetch_dataset(lower_bound, interval, symbol)
    clean_df(df)
    df.to_csv(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Symbol to download", type=str, required=True)
    parser.add_argument("--interval", help="Interval to download", type=str, required=True)
    parser.add_argument("--lower_bound", help="Lower unix timestamp bound", type=int, default=0)
    parser.add_argument("--target", help="Target file", type=str, required=True)

    run(**vars(parser.parse_args()))
