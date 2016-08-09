from web_slinger import web_slinger
import pandas as pd
import numpy as np
import time

class koh(object):
    '''
    To be used in conjunction with other code.
    Gives user more control over methods for easier debugging
    '''
    def __init__(self, url_template, datelist,spidey):
        '''
        Input: url_template for scraping, list of dates to scrape, name of instantiated selenium webdriver to use.

        Fun Fact: koh likes expressive people.
        '''
        self.url_temp = url_template
        self.datelist = datelist
        self.missing_dates = []
        self.spidey = spidey

    def data_stealer(self):
        '''
        Input:
        Output: list of dates where data was failed to be extracted

        Iterates through datelist and appends date to end of url_template and crawls to pages in datelist and attempts to extract data from each page through methods on web_slinger.
        Fun Fact: Not actually "stealing" data, I paid good money for it.
        '''
        for date in self.datelist:
            datestr = '{0}/{1}/{2}'.format(date.day, date.month, date.year)
            durl = self.url_temp+datestr
            try:
                self.spidey.web_crawl(durl)
                time.sleep(2)
                try:
                    self.spidey.extract_data()
                    print 'extracting data for {0}'.format(date)
                except:
                    self.missing_dates.append(date)
                    print 'FAILED to extract data for {0}'.format(date)
            except:
                print 'Could not reach data on {0}'.format(date)

            finally:
                self.spidey.cookie_logger()

        return self.missing_dates
'''
Below are various dataframe filters that are more numerous and explained in datapipe.py
'''

def df_filter(topn,df):
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
    tag = df.groupby('Channel')['Main Game'].unique().reset_index()
    tag['num_games'] = (tag['Main Game']).apply(len)
#     tag = tag[tag['num_games']>=num_games]
    chars = tag['Channel'].values
    tag.set_index('Channel', inplace=True)
    frames = []
    for pers in chars:
        dft = df[df['Channel']==pers]
        dft['num_games'] = tag.loc[pers]['num_games']
        frames.append(dft)
    dfout = pd.concat(frames, ignore_index=True)
    return dfout

def chan_filter(df,channel):
    chan_mask = df['Channel'].str.contains(channel, case=False)
    cdf = df[chan_mask]
#     cdf.set_index('date', inplace=True)
    return cdf
