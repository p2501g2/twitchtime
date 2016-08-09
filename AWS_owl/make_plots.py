import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import os
from goldenkappa import Kappa

def game_filter(df, game):
    '''
    Input: dataframe and game to be filtered (not case sensitive)
    Output: filtered dataframe where main game field contains input string
    While case sensitive discrepencies have been accounted for, sometimes games are listed differently that normal EX: StarCraft 2: Legacy of long sequel names, could be simply SC2. In this case the filter won't include that. If there are known iterations of the game, can input as "StarCraft|SC2"
    '''
    game_mask = df['Main Game'].str.contains(game, case=False)
    gdf = df[game_mask]
    return gdf

def game_lpc(df, ax, leg_size=None):
    '''
    Input: The filtered dataframe and the axes of the subplot to be plotted on.
    Output: Plot with different color dot for each different game a streamer streams.

    '''
    colors = iter(cm.rainbow(np.linspace(0,1, len(df['Main Game'].unique()))))
    for game in df['Main Game'].unique():
        game_mask = df['Main Game'].str.contains(game)
        tst = df[game_mask].copy()
        tst.reset_index(inplace=True)
        ax.plot_date(x=tst['date'] ,y=tst["AVG CCV's"],label=game, color = next(colors))
    rolmean = pd.Series.rolling(df["AVG CCV's"], window=7).mean()
    rolstd = pd.Series.rolling(df["AVG CCV's"], window=7).std()
    for label in ax.get_yticklabels():
        label.set_fontsize(16)
    ax.plot(rolmean, color='red', label='Rolling mean')
    ax.plot(rolstd, color = 'black', label='Rolling std')

def plot_comp(channel1, channel2, save=False):
    '''
    Input: 2 Channels to be compared side-by-side on a shared y-axis plot. calls the game_lpc function above for ploting
    Output: if save=true, will save plot to figures folder

    '''
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(16,10))
    mk.cfilt(channel1)
    df1 = mk.cdf
    mk.cfilt(channel2)
    df2 = mk.cdf
    ax1.tick_params(labelsize=18)
    [tick.set_rotation(45) for tick in ax1.get_xticklabels()]
    ax2.tick_params(labelsize=18)
    [tick.set_rotation(45) for tick in ax2.get_xticklabels()]
    ax1 = game_lpc(df1, ax1)
    ax2 = game_lpc(df2, ax2)
    if save==False:
        plt.show()
    else:
        fig.savefig('figures/G_{0}_{1}.png'.format(channel1, channel2))



def top5plot(df, save=False, ls = None):
    '''
    Input: Dataframe containing independent variables. Doesn't top5 games, uses known top5 that have been discerned using EDA
        Calls the plot_gamesAVG function below on created top5 dataframe
    Output: Plot of top5 games over the 2 years of data
    '''
    hdf = game_filter(df, 'Hearthstone').copy()
    csdf = game_filter(df, 'Counter-strike').copy()
    dotadf = game_filter(df, 'dota').copy()
    loldf = game_filter(df, 'league of legends').copy()
    owdf = game_filter(df, 'overwatch').copy()
    hdf['Main Game'] = 'Hearthstone'
    dotadf['Main Game'] = 'Dota 2'
    loldf['Main Game'] = 'League of Legends'
    owdf['Main Game'] = 'Overwatch'
    csdf['Main Game'] = 'Counter-Strike: Gloval Offensive'
    fdf = pd.concat([hdf, dotadf, loldf, owdf, csdf])
    gav = fdf.groupby(['date','Main Game']).sum().reset_index()
    gav.set_index('date', inplace=True)
    gav.sort_index(inplace=True)
    plot_gamesAVG(gav,ls)
    if save == False:
        plt.show()
    else:
        plt.savefig('figures/top5games.png')

def plot_gamesAVG(df, leg_size=None):
    '''
    Input: Dataframe.
    Plots average viewership for each unique game in the Dataframe.
    Called by top5plot which creates a temporary dataframe with only 5 games
    '''
    colors = iter(cm.rainbow(np.linspace(0,1, len(df['Main Game'].unique()))))
    plt.figure(figsize=(16,14))
    plt.xticks(rotation=45)
    plt.tick_params(labelsize=18)

    for game in df['Main Game'].unique():
        dft = game_filter(df, game)
        plt.plot(dft["AVG CCV's"], label=game, color = next(colors))
    plt.legend(loc='best', prop={'size':leg_size})

if __name__ == '__main__':
    dfg = pickle.load(open('pickle_pile/dfg.pkl','rb'))
    # top5plot(dfg, ls=16)
    # plot_comp('destiny', 'summit1g')
    # cvparams = pickle.load(open('pickle_pile/cross_val_smorc.pkl', 'rb'))

    # top5plot(dfg, save=True, ls=14)

    mk = Kappa('pickle_pile/dff.pkl')
    plot_comp('destiny','summit1g', True)
