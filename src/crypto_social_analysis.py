import pandas as pd
from itertools import product
import json

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker

from tqdm import tqdm

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

def get_coin_list():
    '''Retrieve list of coins from cryptocompare.com
    to get list of coin IDs
    '''
    coin_list_url='https://www.cryptocompare.com/api/data/coinlist/'
    response = urlopen(coin_list_url)
    coin_list_data = json.loads(response.read().decode('utf-8'))
    return coin_list_data

def get_historical_price(date_index, coin_list, fiat_list=['USD']):
    '''Get hisorical cryptocoin prices (per day) cryptocompare.com
    valued in one or more fiats.
    '''
    if not isinstance(coin_list, list):
        if isinstance(coin_list, str):
            coin_list = [coin_list]
        else:
            print('coin_list must be a list, not a {}'.format(type(coin_list)))
            return
    if len(coin_list) > 1:
        print('Can only process one coin at a time. You submitted {}'.format(coin_list))
        return
    
    print('Getting historical price info for {}: {} to {}'.format(
        coin_list,
        date_index[0].strftime('%Y-%m-%d'),
        date_index[-1].strftime('%Y-%m-%d'),
        ))
    
    baseurl = 'https://min-api.cryptocompare.com/data/pricehistorical'

    cols = ['{c}-{f}'.format(c=coin, f=fiat) for coin, fiat in list(product(coin_list, fiat_list))]
    
    df_hist = pd.DataFrame(index=date_index, columns=cols)
    
    for date in tqdm(date_index):
        for coin in coin_list:
            url = '{u}?fsym={c}&tsyms={f}&ts={d}'.format(u=baseurl,
                                                         c=coin,
                                                         f=','.join(fiat_list),
                                                         d=(date.value // 10**9),
                                                         )
            response = urlopen(url)
            data = json.load(response)
            #print(url)
            #print(data)

            for fiat in fiat_list:
                df_hist.loc[date, '{c}-{f}'.format(c=coin, f=fiat)] = data[coin][fiat]
    return df_hist

def traffic_to_df(sr, traffic_dict, time_unit='day'):
    '''Turn traffic stats from subreddit object into a DataFrame
    '''
    date_str = 'date'
    
    if time_unit == 'day':
        cols = ['unique_vis', 'pageviews', 'subscriptions']
    elif time_unit == 'hour':
        cols = ['unique_vis', 'pageviews']
    elif time_unit == 'month':
        cols = ['unique_vis', 'pageviews']
    
    sr_cols = [date_str] + ['{}-{}'.format(sr, col) for col in cols]
    df = pd.DataFrame(data=traffic_dict[time_unit], columns=sr_cols)
    df[date_str]  = pd.to_datetime(df[date_str], unit='s')
    df = df.set_index(date_str)
    df = df.sort_index()
    
    return df

def get_subreddit(coin_list_data, coin):
    # get social stats by ID
    # API documentation https://www.cryptocompare.com/api/#-api-data-socialstats-
    social_stats_url_base = 'https://www.cryptocompare.com/api/data/socialstats/?id='
    
    social_stats_url = '{}{}'.format(social_stats_url_base, coin_list_data['Data'][coin]['Id'])
    
    print('Getting social info for {}'.format(coin))
    response = urlopen(social_stats_url)
    social_stats_data = json.load(response)
    
    sr = str(social_stats_data['Data']['Reddit']['link'].strip('/').rsplit('/', 1)[-1])
    return sr

def get_coin_stats_price(coin_list, coin_list_data, reddit,
                         coin_dict=None,
                         sr_list_str=None,
                         time_unit=None,
                         ):
    
    if coin_dict is None:
        coin_dict = {}
    if sr_list_str is None:
        sr_list_str = 'subreddit_list'
    if time_unit is None:
        time_unit = 'day'
    
    # initialize DataFrame to store info
    df_coins = None
    
    for coin in coin_list:
        if coin not in coin_dict.keys():
            coin_dict[coin] = {}
        coin_dict[coin]['name'] = coin_list_data['Data'][coin]['CoinName']
        if sr_list_str not in coin_dict[coin].keys():
            coin_dict[coin][sr_list_str] = []

        # retrieve official subreddit name
        sr_official = get_subreddit(coin_list_data, coin)
        # add it to the subreddit list
        coin_dict[coin][sr_list_str].append(sr_official)

        for sr in coin_dict[coin][sr_list_str]:
            print('Getting subreddit info for {}: /r/{}'.format(coin, sr))
            subreddit = reddit.subreddit(sr)

            # initialize to store traffic stats
            coin_dict[coin][sr] = {}

            # get traffic stats
            try:
                traffic_dict = subreddit.traffic()
            except:
                print('No traffic information found for {}: /r/{}'.format(coin, sr))
                traffic_dict = {}

            if len(traffic_dict) > 0:
                # if traffic data exists
                coin_dict[coin][sr]['traffic'] = traffic_dict

                df = traffic_to_df(sr, traffic_dict, time_unit=time_unit)

                if df_coins is not None:
                    df_coins = df_coins.merge(df, how='left', left_index=True, right_index=True)
                else:
                    df_coins = df.copy()

        if df_coins is not None:
            date_index = df_coins.index
        else:
            date_index = pd.date_range(pd.to_datetime('today') - pd.Timedelta(56, 'D'), pd.to_datetime('today'))

        df_hist = get_historical_price(date_index, [coin])

        if df_coins is not None:
            df_coins = df_coins.merge(df_hist, how='left', left_index=True, right_index=True)
        else:
            df_coins = df_hist.copy()
    return df_coins

def plot_coin_reddit_price(df_coins, coin_dict, coin,
                           fiat='USD',
                           sr_list_str=None,
                           subreddit_list=None,
                           figsize=(12, 8),
                           nticks=11,
                           ):
    '''Plot coin price and subreddit traffic stats
    '''
    if sr_list_str is None:
        sr_list_str = 'subreddit_list'
    
    if subreddit_list is None:
        subreddit_list = coin_dict[coin][sr_list_str]
    
    for sr in subreddit_list:
        fig, ax1 = plt.subplots(figsize=figsize)

        lines = []
        labels = []

        # plot price
        df_coins['{}-{}'.format(coin, fiat)].plot(ax=ax1,
                                                  color='k',
                                                  label='{} price ({})'.format(coin, fiat),
                                                  )

        ax1.set_title('{}, /r/{}'.format(coin_dict[coin]['name'].capitalize(), sr));

        ax1.set_xlabel('Date')
        ax1.set_ylabel('{} price ({})'.format(coin, fiat))
        ax1.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines.extend(lines1)
        labels.extend(labels1)

        if len(df_coins.filter(like=sr).columns.tolist()) > 0:
            ax2 = ax1.twinx()

            if '{}-unique_vis'.format(sr) in df_coins.columns:
                df_coins['{}-unique_vis'.format(sr)].plot(ax=ax2,
                                                          color='b',
                                                          label='/r Unique Visitors',
                                                          #secondary_y=True,
                                                          )
            if '{}-pageviews'.format(sr) in df_coins.columns:
                df_coins['{}-pageviews'.format(sr)].plot(ax=ax2,
                                                         color='r',
                                                         label='/r Page Views',
                                                         #secondary_y=True,
                                                         )
            ax2.set_xlabel('')
            ax2.set_ylabel('Visitors & Page Views')
            ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))

            lines2, labels2 = ax2.get_legend_handles_labels()
            lines.extend(lines2)
            labels.extend(labels2)

            if '{}-subscriptions'.format(sr) in df_coins.columns:
                ax3 = ax1.twinx()
                rspine = ax3.spines['right']
                rspine.set_position(('axes', 1.1))
                ax3.set_frame_on(True)
                ax3.patch.set_visible(False)
                fig.subplots_adjust(right=0.75)

                df_coins['{}-subscriptions'.format(sr)].plot(ax=ax3,
                                                             color='g',
                                                             label='/r New Subscriptions',
                                                             )
                ax3.set_xlabel('')
                ax3.set_ylabel('New Subscriptions')
                ax3.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))

                lines3, labels3 = ax3.get_legend_handles_labels()
                lines.extend(lines3)
                labels.extend(labels3)

        ax1.legend(
            lines,
            labels,
            loc='best');

