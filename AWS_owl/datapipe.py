import cPickle as pickle
import pandas as pd
import numpy as np
import os

'''
Further development could involve creating a datapipe object for further OOP'ing.
'''


def tdeval(df):
    '''
    Input: dataframe with date column
    Output: dataframe with new column containing number of days since last stream.
    '''
    channels = df['Channel'].unique()
    frames = []
    for s in channels:
        dft = df[df['Channel']==s].copy()
        dft.sort_values('date', inplace=True)
        dft['tdelta'] = (dft['date']-dft['date'].shift()).fillna(0)
        frames.append(dft)
    dfout = pd.concat(frames, ignore_index = True)
    return dfout


def frequency_eval(df):
    '''
    Input: Dataframe with timedelta column.
    Output: Dataframe with new column that has average timedelta between streams for a given channel.

    Primarily used for EDA and finding meaningful separations/clusters of channels.
    '''
    channels = df['Channel'].unique()
    frames = []
    for s in channels:
        dft = df[df['Channel']==s].copy()
        avf = dft['Days_since_last_stream'].mean()
        dft['avg_frequency'] = avf
        frames.append(dft)
    dfout = pd.concat(frames, ignore_index=True)
    #recomended cutoff <=12
    return dfout

def initial_clean(df):
    '''
    Input: dataframe for initial cleaning
    Output: cleaned dataframe
    '''
    df['Language'].fillna('english', inplace=True)
    df['Main Game'].fillna('None', inplace=True)
    df['Channel'].fillna('Unkown', inplace=True)
    df.sort_values(['date','#'], inplace=True)
    df.reset_index(inplace=True)
    return df

def df_filter(topn,df):
    '''
    Input: Average daily stream rank within topn, dataframe to be filtered. Ranked by Hours Watched.
    Output: Filtered dataframe of only streams within topn.
    '''
    #returns data frame with only channels who average in topn of daily streams
    dfrn = df.groupby('Channel')['#'].mean().reset_index()
    dfrn = dfrn[dfrn['#']<=topn]
    topnlist = dfrn['Channel'].values
    frames = []
    for s in topnlist:
        dft = df[df['Channel']==s]
        frames.append(dft)
    dfout = pd.concat(frames, ignore_index = True)
    return dfout

def nstreams_filter(min_num, df):
    '''
    Input: Minimum number of streams, dataframe.
    Output: Filtered dataframe.

    Helpful in filtering for "high volume streamers." Currently slit on 600 which was determined in EDA.
    '''
    tag = df.groupby('Channel')['#'].count().reset_index()
    tag = tag[tag['#']>= min_num]
    hivol = tag['Channel'].values
    frames = []
    for s in hivol:
        dft = df[df['Channel']==s]
        frames.append(dft)
    dfout = pd.concat(frames, ignore_index=True)
    return dfout

def monsterlf_filter(cfreq, df, min_viewer=0):
    '''
    Input: critical streaming frequency for split, dataframe, minimum average viewercount
    Output: filtered dataframe

    Used to find "low volume, high viewer" streams indicative of large events. EX: evo, dreamhack, mlg, ect.
    Best number to split on was 12 days. This was because many events stream for multiple days at a time during the course of an event, droping their overall average. Additionally, 12 seems to be a good cutoff to filter out most "high volume" streamers who take periodic breaks, or prolonged vacations which drives up their average timedelta.
    '''
    tag = df.groupby('Channel')[['avg_frequency', "AVG CCV's"]].mean().reset_index()
    tag = tag[(tag['avg_frequency'] > cfreq) & (tag["AVG CCV's"] >= min_viewer)]
    bigns = tag['Channel'].values
    frames = []
    for s in bigns:
        dft = df[df['Channel']==s]
        frames.append(dft)
    dfout = pd.concat(frames, ignore_index=True)
    return dfout

def add_ngames_col(df):
    '''
    Input: dataframe
    Output: dataframe with new column that has the total number of games streamed for a given channel. Helpful in determing "personality" streamers; as opposed to game-specific streamers.
    '''
    tag = df.groupby('Channel')['Main Game'].unique().reset_index()
    tag['num_games'] = (tag['Main Game']).apply(len)
#     tag = tag[tag['num_games']>=num_games]
    chars = tag['Channel'].values
    tag.set_index('Channel', inplace=True)
    frames = []
    for pers in chars:
        dft = df[df['Channel']==pers].copy()
        dft['num_games'] = tag.loc[pers]['num_games']
        frames.append(dft)
    dfout = pd.concat(frames, ignore_index=True)
    return dfout


def flush_pipe(df):
    '''
    Input: dataframe
    Output: cleaned dataframe

    Sends dataframe through proper cleaning and column adding steps in correct order to create dataframe used for analysis.
    '''
    df = initial_clean(df)
    df = tdeval(df)
    df['Days_since_last_stream'] = (df['tdelta']/np.timedelta64(1,'D')).astype(int)
    df = frequency_eval(df)
    df = add_ngames_col(df)
    df.set_index('date', inplace=True)
    df['dayofweek'] = df.index.dayofweek
    df['weekofyear'] = df.index.weekofyear
    # df.reset_index(inplace=True)
    return df
