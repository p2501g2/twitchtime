import pandas as pd
import os
import cPickle as pickle

def parse_dchan_file(fname):
    datestmp = pd.to_datetime(fname[33:-14], format='%d-%m-%Y')
    dft = pd.read_excel(fname)#, index_col=0)
    dft['date'] = datestmp
    # dft = dft.drop('#', axis = 1)
    return dft

def create_master_df(data_parent_dir, df_pickle_name):
    '''
    Input: Folder name of parent directory for data, name of dataframe file to be saved
    Output: pickled dataframe saved as df_pickle_name combining all of the data tables
    Notes:
    Only works because all files in dir are excel spreadsheet data
    Can be read later with df = pd.read_pickle(df_pickle_name)
    '''
    df = pd.DataFrame()
    for fname in os.listdir(data_parent_dir):
        f = data_parent_dir+fname
        dft = parse_dchan_file(f)
        df = pd.concat([df,dft], ignore_index=True)
    df.to_pickle('pickle_pile/'+df_pickle_name) #Puts in pickle_pile folder
    #DEPRECATED: using dfbuild instead

def dfbuild(data_parent_dir, df_pickle_name):
    frames = [parse_dchan_file(data_parent_dir+f) for f in os.listdir(data_parent_dir)]
    df = pd.concat(frames, ignore_index = True)
    df.to_pickle('pickle_pile/'+df_pickle_name)

def list_comp(l1,l2):
    '''
    l1 = ideal date_range
    l2 = unique dates in dataframe
    '''
    miss = []
    for i in l1:
        miss.append(i not in l2)
    return miss

def miss_list(daterange, miss_mask):
    return [i for i, flag in zip(daterange,miss_mask) if flag==1]

def cocoon(date_col, fname):
    dr = list(pd.date_range(date_col.min(), date_col.max()))
    d = [pd.to_datetime(i) for i in date_col]
    miss_mask = list_comp(dr,d)
    missedlist = miss_list(dr,miss_mask)
    pickle.dump(missedlist, open(fname,'wb'))

#call output with::: junk = pickle.load(open('pickle_pile/savemiss.pkl','rb'))

if __name__ == '__main__':
        # create_master_df('xcdata/', 'testd.pkl')
        #for testing
        # create_master_df('xcdata/', 'day_chan_data.pkl')
        dfbuild('xcdata/', 'cdad.pkl')
