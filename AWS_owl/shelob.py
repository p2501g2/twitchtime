from web_slinger import web_slinger
import pandas as pd
import numpy as np
import time


def export_channel_date_range(datelist):
    '''
    Input: A list of dates for which daily channel data will be extracted
    Output: A missing_dates list. Can be used as input again.
    Some testing makes me think that some of the pages are just problamatic.

    Initial run from 2014/10/8 -> 2015/4/8
    Missing Dates -> 2015/03/29
                     2015/03/30
                     2015/04/06
    Iterated again over missing_dates
    Got data for 2015/04/06
    Still missing -> 2015/03/29
                     2015/03/30
    Can't Access these pages through any methods. Bad pages.
    Moving forward, Iterate through missing_dates a couple times
    Hopefully it shrinks considerably each time. save final list object somewhere


    '''
    spidey = web_slinger()
    spidey.spin_web()
    spidey.login()
    channel_url_template = 'https://gamoloco.com/scoreboard/channels/daily/'
    time.sleep(3)
    # spidey.logged_in_check()
    missing_dates = []
    if spidey.logged_in == False:
        try:
            spidey.login()
        except:
            print 'retry login error'
    else:
        print 'login successful'

    for date in datelist:
        datestr = '{0}/{1}/{2}'.format(date.day, date.month, date.year)
        chan_data_url = channel_url_template+datestr
        try:
            spidey.web_crawl(chan_data_url)
            time.sleep(2)
            # spidey.driver.save_screenshot('{0}.png'.format(date))
            try:
                spidey.extract_data()
                print 'extracting'
                time.sleep(2)
            except:
                missing_dates.append(date)
                print 'failed to extract data for {0}'.format(date)
        except:
            print 'could not reach data on {0}'.format(date)
        finally:
            spidey.logged_in_check()

    spidey.driver.close()

    return missing_dates

if __name__ == '__main__':
    datelist = pd.date_range(pd.datetime(2014,10,8),pd.datetime(2015,4,8))
    # missing_dates = export_channel_date_range(datelist)
    # minidate = datelist[:7]

    #small testing date list
#     spidey = web_slinger()
#     #instantiate
#     spidey.spin_web()
#     # spidey.logged_in_check()
#     spidey.login_to_web()
#     channel_url_template = 'https://gamoloco.com/scoreboard/channels/daily/'
#     # print channel_url_template+'/{0}/{1}/{2}'.format(minidate[0].day, minidate[0].month, minidate[0].year)
#
#     # spidey.logged_in_check()
# # data_url = 'https://gamoloco.com/scoreboard/channels/daily/{0}/{1}/{2}'.format(day,month,year)
#     time.sleep(3)
#     spidey.logged_in_check()
#     missing_dates = []
#     if spidey.logged_in == False:
#         try:
#             spidey.login_to_web()
#         except:
#             print 'retry login error'
#     else:
#         print 'login successful'
#
#     for date in minidate:
#         datestr = '{0}/{1}/{2}'.format(date.day, date.month, date.year)
#         chan_data_url = channel_url_template+datestr
#         try:
#             time.sleep(2)
#             spidey.web_crawl(chan_data_url)
#             try:
#                 time.sleep(2)
#                 spidey.extract_data()
#                 print 'extracting'
#             except:
#                 missing_dates.append(date)
#                 print 'failed to extract data for {0}'.format(date)
#         except:
#             print 'could not reach data on {0}'.format(date)
#         finally:
#             spidey.logged_in_check()
