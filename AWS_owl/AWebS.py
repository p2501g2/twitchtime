from shelob import export_channel_date_range
from web_slinger import web_slinger
import pandas as pd
import numpy as np
import time
from pyvirtualdisplay import Display
import multiprocessing
from koh import koh
import cPickle as pickle
import os
from datapipe import flush_pipe

'''
AWebS is to be run on AWS EC2 instance. Will utilize the function from shelob
to iterate through all the dates.

Suggested methods for display stuff
http://stackoverflow.com/questions/6183276/how-do-i-run-selenium-in-xvfb
https://dzone.com/articles/taking-browser-screenshots-no

run
$ sudo apt-get install python-pip xvfb xserver-xephyr
$ sudo pip install selenium
'''

class Wanshitong(object):
    '''
    Fun Fact: Translates to "He who knows 10,000 things" in Chinese

    generally called as:
    owl = Wanshitong(path/to/data/folder, datelist)
    '''

    def __init__(self, data_parent_dir, initial_datelist=None):
        '''
        Input: directory of data destination, location, list of dates to be scraped
        Output: Instatiated selenium webdirver and Wanshitong object

        owl_anger_lvl corresponds to how many times the datelist has been iterated over. So when a page is missed, it will be attempted to be scraped again on the next iteration.


        '''
        self.spidey = web_slinger()
        self.datelist = initial_datelist
        self.display = Display(visible=0, size=(1280,1024))
        self.data_parent_dir  = data_parent_dir
        self.owl_anger_lvl = 0
        # self.missing_dates = []

    def url_template(self, genre):
        '''
        Input: genre = 'channels' OR 'games'
        Output: url template to be scraped without dates; which are added manually.
        '''
        self.url_temp = 'https://gamoloco.com/scoreboard/'+genre+'/daily/'

    def parse_dchan_file(self, fname):
        '''
        Input: filename of scraped excel spreadsheat to be parsed.
        Output: dataframe with date column added

        Numbers used to extract datetime represent where the date sits in the filename string. the -14 is always the same as the file contains 14 characters after the date. the 36 is determined from the number of characters in the parent directory plus the number of characters in url_template. The 36 number must be tuned specifically.
        '''
        datestmp = pd.to_datetime(fname[36:-14], format='%d-%m-%Y')
        dft = pd.read_excel(fname)#, index_col=0)
        dft['date'] = datestmp
        # dft = dft.drop('#', axis = 1)
        return dft

    def dfbuild(self, df_pickle_name):
        '''
        Input: name of dataframe to be pickled
        Output: saves pickled dataframe to pickle folder, sets object's dataframe equal to built dataframe
        filename does not require "pickle_pile/" as it is added automatically to file name to ensure proper destination of pickled dataframe
        '''
        # frames = [self.parse_dchan_file(self.data_parent_dir+f) for f in os.listdir(self.data_parent_dir)]
        frames = [self.parse_dchan_file(self.data_parent_dir+f) for f in os.listdir(self.data_parent_dir) if f.endswith('.xlsx')]
        df = pd.concat(frames, ignore_index = True)
        df.to_pickle('pickle_pile/'+df_pickle_name)
        self.df = df

    def _list_comp(self, l1,l2):
        '''
        Input:
        l1 = ideal date_range
        l2 = unique dates in dataframe
        Output: Missed mask for filtering only dates unsuccessfully scraped.

        Generally called by _miss_list with this output as miss_mask
        Used for finding dates that were not successfully scraped.
        '''
        miss = []
        for i in l1:
            miss.append(i not in l2)
        return miss

    def _miss_list(self, daterange, miss_mask):
        '''
        Input: ideal daterange, mask for finding misses
        Output: List of dates failed to be extracted. Additional scraping procedures use this list for next iteration, angering the owl.
        '''
        return [i for i, flag in zip(daterange,miss_mask) if flag==1]

    def _cocoon(self, date_col, fname):
        '''
        Input: date columns of dataframe with dates of extracted files, filename of pickled miss list for further use.
        Output: pickled list of missed dates to be iterated.

        '''
        dr = list(pd.date_range(date_col.min(), date_col.max()))
        d = [pd.to_datetime(i) for i in date_col]
        miss_mask = self._list_comp(dr,d)
        missedlist = self._miss_list(dr,miss_mask)
        pickle.dump(missedlist, open(fname,'wb'))

    def get_miss_list(self, df_pickle_name, pklmiss_fname):
        '''
        Input: pickled dataframe filename, pickled missed date list filename.
        Output: list of missed dates, updated df attribute of Wanshitong
        '''
        self.dfbuild(df_pickle_name)
        dft = pickle.load(open('pickle_pile/'+df_pickle_name))
        self.cocoon(dft['date'],pklmiss_fname)
        miss_dates = pickle.load(open(pklmiss_fname))
        # return miss_dates
        # self.missed_date_check = miss_dates
        return miss_dates

    def assimilate(self, datelist):
        '''
        Input: List of dates to be scraped.
        Output: sets missed_dates attribute to missed dates

        Instantiates and calls a koh object which operates the web scraper.
        Fun Fact: Koh is an ancient spirit that steals faces from a popular tv show. Called koh because it 'steals the face' of the web_slinger, as the spidey object is operated and controlled by koh.
        '''
        face_stealer = koh(self.url_temp, datelist, self.spidey)
        print 'run get_miss_list(params) to compute list of missed dates for further iteration'
        self.missed_dates =  face_stealer.data_stealer()

    def send_foxes(self):
        '''
        Input:
        Output: print statements with further instructions to start scraping.
        Website proved to be very finicky so scraping is manually commenced after a successful login.

        Calls methods of web_slinger, the web scraping object, instantiated as spidey. Creats webdriver firefox profile and attempts login.
        '''
        # self.display.start()
        self.spidey.spin_web()
        self.spidey.login()
        self.spidey.cookie_logger()
        time.sleep(1)
        try:
            self.spidey._click_login_submitter()
        except:
            print 'tried clicking from spirit library'
        time.sleep(1)
        self.spidey.cookie_logger()
        if self.spidey.logged_in == False:
            print 'appears login failed'
            print 'run command spidey.login()'
            print 'check if logged in with spidey.cookie_logger()'
        else:
            print 'appear to be logged in'
            print 'begin date extraction by running, >> owl.knowledge_seekers(iternum=#)'

    def knowledge_seekers(self, iternum=1):
        '''
        Input: Number of iterations through datelist for data extraction
        Output: list of missed dates for each iteration

        Begins actual webscraping process by calling the assimilate function on the most recent list of dates, which is appended with a new list of missed dates after each iteration.
        Fun Fact: Foxes sometimes aren't the best knowledge seekers which is why they require some oversight. When they fail to bring back data fro all the dates this angers the owl.
        '''
        i = 0
        self.super_miss_list = []
        self.super_miss_list.append(self.datelist)
        while i < iternum:
            i += 1
            self.assimilate(self.super_miss_list[-1])
            miss_iter = self.get_miss_list('df{0}.pkl'.format(i), 'pickle_pile/pklmiss{0}.pkl'.format(i))
            self.super_miss_list.append(miss_iter)

        print 'knowledge seekers have returned'
        print 'if their information is unsatisfactory can run again with: >> big_owl_mad()'

    def big_owl_mad(self):
        '''
        Input:
        Output: Continued date iteration

        Can be called if the preset number of iterations was not sufficient to extract dates.
        Main reason this is a separate function is because usually the list of dates is VERY small at this point, usually there is something inherantly wrong with the data on that page, which could be investigated manually. This way an empty page or one missing a data table will not be iterated on over and over due to always being on the missed dates list.

        '''
        self.owl_anger_lvl += 1
        self.assimilate(self.super_miss_list[-1])
        miss_iter = self.get_miss_list('dfAowl{0}.pkl'.format(self.owl_anger_lvl), 'pickle_pile/A_owl_miss{0}.pkl'.format(self.owl_anger_lvl))
        self.super_miss_list.append(miss_iter)
        print 'if satisfied >> bury_library()'

    def bury_library(self):
        '''
        Input:
        Output: Quits and closes instantiated web scraping objects like virtual display and webdriver

        Fun Fact: Wanshitong buried his library

        '''
        self.display.stop()
        self.spidey.driver.quit()

    def flush(self, df):
        '''
        Input: uncleaned dataframe
        Output: cleaned dataframe

        Calls flush_pipe function in the datapipe.py file
        '''
        return flush_pipe(df)


if __name__=='__main__':
    datelist = pd.date_range(pd.datetime(2016,7,19), pd.datetime(2016,7,31))
    owl = Wanshitong(datelist, '~/downloads')
    owl.send_foxes()
    owl.url_template('channels')

    # display = Display(visible=0, size=(1280,1024))
    # display.start()
    # spidey = web_slinger()
    # spidey.spin_web()
    # spidey.login()
    # spidey.cookie_logger()
    # time.sleep(1)
    # if spidey.logged_in == False:
    #     print 'appears login failed'
    #     print 'run command spidey.login()'
    #     print 'check if logged in with spidey.cookie_logger()'
    # else:
    #     print 'appear to be logged in'
    #     print 'begin date extraction by running, >>missedi = assimilate(url_temp,datelist,spidey)'
    # url_temp = url_template('channels')
    # missing_dates = []




    # display.stop()


# def url_template(genre):
#     '''
#     genre = 'channels' OR 'games'
#     '''
#     return 'https://gamoloco.com/scoreboard/'+genre+'/daily/'

# def assimilate(url_temp, datelist, spidey):
#     face_stealer = koh(url_temp, datelist, spidey)
#     print 'run get_miss_list(params) to compute list of missed dates for further iteration'
#     return face_stealer.data_stealer()

# def parse_dchan_file(fname):
#     datestmp = pd.to_datetime(fname[33:-14], format='%d-%m-%Y')
#     dft = pd.read_excel(fname)#, index_col=0)
#     dft['date'] = datestmp
#     # dft = dft.drop('#', axis = 1)
#     return dft
#
#
# def dfbuild(data_parent_dir, df_pickle_name):
#     frames = [parse_dchan_file(data_parent_dir+f) for f in os.listdir(data_parent_dir)]
#     df = pd.concat(frames, ignore_index = True)
#     df.to_pickle('pickle_pile/'+df_pickle_name)
#
# def list_comp(l1,l2):
#     '''
#     l1 = ideal date_range
#     l2 = unique dates in dataframe
#     '''
#     miss = []
#     for i in l1:
#         miss.append(i not in l2)
#     return miss
#
# def miss_list(daterange, miss_mask):
#     return [i for i, flag in zip(daterange,miss_mask) if flag==1]
#
# def cocoon(date_col, fname):
#     dr = list(pd.date_range(date_col.min(), date_col.max()))
#     d = [pd.to_datetime(i) for i in date_col]
#     miss_mask = list_comp(dr,d)
#     missedlist = miss_list(dr,miss_mask)
#     pickle.dump(missedlist, open(fname,'wb'))
#
# def get_miss_list(data_parent_dir, df_pickle_name, pklmiss_fname):
#     dfbuild(data_parent_dir, df_pickle_name)
#     dft = pickle.load(open('pickle_pile/'+df_pickle_name))
#     cocoon(dft['date'],pklmiss_fname)
#     miss_dates = pickle.load(open(pklmiss_fname))
#     return miss_dates







    #do stuff
    # datelist = pd.date_range(pd.datetime(2014,8,4),pd.datetime(2016,7,18))
    # missing_dates = export_channel_date_range(datelist)
    # missing2 = export_channel_date_range(missing_dates)
    # missing3 = export_channel_date_range(missing2)
