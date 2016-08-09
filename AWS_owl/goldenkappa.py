import cPickle as pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from koh import nstreams_filter, add_ngames_col, chan_filter
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.grid_search import GridSearchCV
import itertools
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
rforecast = rpackages.importr('forecast')


class Kappa(object):
    def __init__(self, fname, cvparams=True):
        '''
        Input: file path to dataframe to be used for analysis
        Output: Kappa object with cleaned dataframe
        '''
        self.fname = fname
        temp = pickle.load(open(fname))
        self.df = add_ngames_col(nstreams_filter(600,temp))
        self.df.set_index('date', inplace=True)
        self.df['dayofweek'] = self.df.index.dayofweek
        self.df['weekofyear'] = self.df.index.weekofyear
        self.df['year'] = self.df.index.year
        # self.rfr_params = {'n_estimators':300,
        #       'max_features':'sqrt',
        #       'n_jobs':-1}

        if cvparams==True:
            self.cvparams = pickle.load(open('pickle_pile/cross_val_smorc.pkl','rb'))
        self.cv_pred = {}

    def cfilt(self,channel):
        '''
        Input: channel to filter for
        Output: channel-specific dataframe
        '''
        self.cdf=chan_filter(self.df, channel)
        self.cdf.sort_index(inplace=True)


    def dumb_set(self,dft):
        '''
        Input: Initial dataframe
        Output: X, y for machine learning algorithms with dummified categorical variables.
        '''
        # for i in np.arange(7):
        #     dft['lagday{0}'.format(i+1)] = (dft["AVG CCV's"].shift((i+1))).fillna(0)
        y = dft["AVG CCV's"]
        dc = ['AirTime','Platform', 'tdelta','avg_frequency','Language', 'index','#', "AVG CCV's", "Max CCV's", 'Hours Watched']# 'Hours Watched', 'Channel', 'Main Game'

        X = dft.drop(dc, axis=1)
        X = pd.get_dummies(X, columns = ['Channel', 'Main Game', 'dayofweek'])
        #eventually add Language
        return X, y


    def _make_holdout_split(self, df, leaveout=3): #
        '''
        Input: dataframe and # leaveout weeks.
        Output: X,y training and hold data partitions
        '''
        # self.folds = pd
        lod = leaveout*7
        start, end = df.index.min(), df.index.max()
        self.folds = pd.date_range(start, end, freq='7D')#'{0}D'.format(lod))
        self.Xset, self.yset = self.dumb_set(df)
        lo = self.folds[-leaveout:][0]
        X_trainset = self.Xset.query('date < @lo')
        y_trainset = self.yset.reset_index().query('date < @lo')
        X_holdset = self.Xset.query('date >= @lo')
        y_holdset = self.yset.reset_index().query('date >= @lo')
        self.X_trainset = X_trainset.copy()
        self.y_trainset = y_trainset.copy().set_index('date')
        self.X_holdset  = X_holdset.copy()
        self.y_holdset = y_holdset.copy().set_index('date')
        self.X_train = self.X_trainset.reset_index().copy()
        self.y_train = self.y_trainset.reset_index().copy()
        self.X_hold = self.X_holdset.reset_index().copy()
        self.y_hold = self.y_holdset.reset_index().copy()

    def _fchain_kfold_indicies(self, lag=1, ahead=1):
        '''
        Input: lag weeks, ahead weeks
        Output: forward chain kfold cross validation indices for time series.
        '''
        #currently avoiding dummy problems by dummifying early, need to fix
        # and adapt later down the road. Also add cols
        ld = pd.Timedelta(days=lag*7)
        ad = pd.Timedelta(days=ahead*7)
        kstart, kend = self.X_trainset.index.min(), self.X_trainset.index.max()
        period = lag*7 + ahead*7
        self.kfolds = pd.date_range(kstart,kend, freq='{0}D'.format(period))
        self.train_kfoldi = []
        self.test_kfoldi = []
        self.fkfoldi = []
        # self.kfoldxi = []
        # self.kfoldyi = []
        for i, f in enumerate(self.kfolds):
            j = 1+i
            if f==kstart:
                #For first fold
                udb = self.kfolds[1] - ad
                train_xset = self.X_train.query('date < @udb')
                train_yset = self.y_train.query('date < @udb')
                test_yset = self.y_train.query('date >= @udb & date < @self.kfolds[1]')
                test_xset = self.X_train.query('date >= @udb & date < @self.kfolds[1]')

            elif i == len(self.kfolds)-1:
                #For last fold
                udb = kend - ad
                train_yset = self.y_train.query('date < @udb')
                train_xset = self.X_train.query('date < @udb')
                test_xset = self.X_train.query('date >= @udb')
                test_yset = self.y_train.query('date >= @udb')

            else:
                #middle folds
                udb = self.kfolds[j]-ad
                train_xset = self.X_train.query('date < @udb')
                train_yset = self.y_train.query('date < @udb')
                test_xset = self.X_train.query('date >= @udb & date < @self.kfolds[@j]')
                test_yset = self.y_train.query('date >= @udb & date < @self.kfolds[@j]')

            self.testx_ind = test_xset.index.values
            self.testy_ind = test_yset.index.values
            self.trainx_ind = train_xset.index.values
            self.trainy_ind = train_yset.index.values
            self.train_kfoldi.append([self.trainx_ind, self.trainy_ind])
            self.test_kfoldi.append([self.testx_ind, self.testy_ind])
            self.fkfoldi.append((self.trainx_ind, self.testx_ind))

        self.Xtrn = self.X_train.drop('date', axis=1)
        self.ytrn = self.y_train.drop('date', axis=1)
        # self.Xhld = self.X_hold.drop('date', axis=1)
        # self.yhld = self.yhld.drop('date', axis=1)



    def run_cvmod(self, channel):
        '''
        DEPRECATED:
        Originally used for testing/debugging
        '''
        self.cfilt(channel)
        # self.cdf.sort_index(inplace=True)
        self._make_holdout_split(mk.cdf)
        self._fchain_kfold_indicies()
        # self.Xtrn = self.X_train.drop('date', axis=1)
        # self.ytrn = self.y_train.drop('date', axis=1)
        self.rfrcv = RandomForestRegressor(**self.rfr_params)
        mses = []
        r2s = []
        for j in xrange(len(self.kfolds)):
            print '******'
            print 'Evaluating Fold #{0}'.format(j)
            print '******'
            Xtrain_indices, ytrain_indices = self.train_kfoldi[j]
            Xtest_indices, ytest_indices = self.test_kfoldi[j]
            xtrain = self.Xtrn.iloc[Xtrain_indices]
            ytrain = self.ytrn.iloc[ytrain_indices]
            xtest = self.Xtrn.iloc[Xtest_indices]
            ytest = self.ytrn.iloc[ytest_indices]
            self.rfrcv.fit(xtrain.values, ytrain.values)
            ypred = self.rfrcv.predict(xtest)
            mses.append(mean_squared_error(ytest.values, ypred))
            r2s.append(r2_score(ytest.values, ypred))
        self.rmses, self.r2_scores = np.sqrt(mses), np.array(r2s)

    def test_stationarity(self,channel):
        '''
        Input: channel for testing
        Output: Results of Dickey-Fuller test and plot with data, rolling mean, and rolling std.
        '''
        #requires date index
        ts = chan_filter(self.df, channel)
        ts.sort_index(inplace=True)
        timeseries = ts["AVG CCV's"]
        rolmean = pd.Series.rolling(timeseries, window=7).mean()
        rolstd = pd.Series.rolling(timeseries, window=7).std()
        orig = plt.plot(timeseries, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color = 'black', label='Rolling std')
        plt.legend(loc='best')
        plt.show(block=False)
        print 'Results of Dickey-Fuller Test'
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print dfoutput

    def plot_acf_pacf(self, channel, lags=20):
        '''
        Input: channel and #lags to include
        Output: Plots with autocorrelation function and partial autocorrelation function.
        '''
        #set indexto date in input
        ts = chan_filter(self.df, channel)
        ts.sort_index(inplace=True)
        data = ts["AVG CCV's"]
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        fig = plot_acf(data, lags=lags, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = plot_pacf(data, lags=lags, ax=ax2)
        plt.show()


    def pltrange(self):#, indx_range=None):
        '''
        DEPRECATED:
        Used for early EDA, data-viz and results analysis
        '''
        ypred = self.rfrcv.predict(self.Xtrn)
        # if indx_range != None:
        #     plt.plot(ypred[indx_range])
        #     plt.plot(self.ytrn.values[indx_range])
        #     plt.show(block=False)
        # else:
        plt.plot(ypred)
        plt.plot(self.ytrn.values)
        plt.show(block=False)

    def run_grid_search(self, estimator):
        '''
        Input: estimator name
        Output: best parameters for a given estimator
        '''

        if estimator.__class__.__name__ == 'RandomForestRegressor':
            self.gridsearch = GridSearchCV(estimator,
                                            self.rfr_gsparams,
                                            n_jobs=-1,
                                            verbose=True,
                                            scoring='mean_squared_error',
                                            cv=self.fkfoldi)


            #have this functionr return best params then pass those as argument to cross_val_score and loop through different channels
        elif estimator.__class__.__name__ == 'GradientBoostingRegressor':
            self.gridsearch = GridSearchCV(estimator,
                                            self.gboostR_gsparams,
                                            n_jobs=-1,
                                            verbose=True,
                                            scoring='mean_squared_error',
                                            cv=self.fkfoldi)
        self.gridsearch.fit(self.Xtrn, self.ytrn)
        print self.gridsearch.best_params_
        print 'for ', estimator.__class__.__name__

    def eval_models(self, channel):#deprecated, cvpredict needs partitions
        '''
        DEPRECATED:
        cvpredict requires full partitions of cross val indices. Was attempting to simplify code however forward chain cross val not compatible with cvpredict.
        '''
        self.cfilt(channel)
        self._make_holdout_split(self.cdf)
        self._fchain_kfold_indicies()
        lassoCV_params = {'cv': self.fkfoldi,
                            'n_jobs':-1,
                            'alphas':np.logspace(-4,2,100)}
        ridgeCV_params =  {'cv': self.fkfoldi,
                            'alphas':np.logspace(-4,2,100),
                            'scoring':'mean_squared_error'}
        models = [RandomForestRegressor(**self.cvparams[channel]['RandomForestRegressor']['params']),
        GradientBoostingRegressor(**self.cvparams[channel]['GradientBoostingRegressor']['params']),
        LassoCV(**lassoCV_params),
        RidgeCV(**ridgeCV_params)]
        self.cv_pred[channel] = {}
        for mod in models:
            self.cv_pred[channel][mod.__class__.__name__] = cross_val_predict(estimator=mod, X=self.Xtrn, y=self.ytrn, cv=self.fkfoldi, n_jobs=-1)

        pickle.dump(self.cv_pred[channel], open('pickle_pile/{0}_cvpred.pkl'.format(channel),'wb'))

    def linear_kappa_search(self):
        '''
        DEPRECATED:
        more efficient method used elsewhere.
        Input:
        Output:
        '''
        channels = ['lirik', 'summit1g', 'imaqtpie', 'nl_kripp', 'destiny']
        self.lcvp = {}
        for channel in channels:
            self.lcvp[channel] = {}
            self.cfilt(channel)
            self._make_holdout_split(self.cdf)
            self._fchain_kfold_indicies()
            lassoCV_params = {'cv': self.fkfoldi,
                                'n_jobs':-1,
                                'alphas':np.logspace(-4,2,100)}
            ridgeCV_params =  {'cv': self.fkfoldi,
                                'alphas':np.logspace(-4,2,100),
                                'scoring':'mean_squared_error'}
            models = [LassoCV(**lassoCV_params), RidgeCV(**ridgeCV_params)]
            for regression in models:
                reg_name = regression.__class__.__name__
                self.lcvp[channel][reg_name] = {}
                regression.fit(self.Xtrn, self.ytrn)
                self.lcvp[channel][reg_name][alpha] = regression.alpha_
                mse_scores = cross_val_score(estimator=regression, X=self.Xtrn, y=self.ytrn, scoring='mean_squared_error', n_jobs=-1)
                self.lcvp[channel][reg_name][rmse] = np.sqrt(mse_score).mean()



    def kappa_search(self, channel, estimator):
        '''
        Input: Channel for which to optimize model, estimator for model.
        Output: Gridsearch on estimator using parameters defined in function.
        '''
        self.cfilt(channel)
        self._make_holdout_split(self.cdf)
        self._fchain_kfold_indicies()
        self.rfr_gsparams = {'n_estimators': [10, 100, 200, 300],
                                'criterion': ['mse'],
                                'min_samples_split': [2,4,6],
                                'min_samples_leaf': [1,2],
                                'max_features': ['sqrt', None, 'log2'],
                                'n_jobs':[-1]
                                }
        self.gboostR_gsparams = {'loss': ['ls','lad','huber'],
                                    'learning_rate': [.001, .01, .1, 1, 2],
                                    'n_estimators': [50, 100, 200],
                                    'max_depth': [2,5,8,10],
                                    'max_features': [None,'sqrt','log2']
                            }
        self.xts = self.X_train.set_index('date')
        self.yts = self.y_train.set_index('date')
        self.arima_params = {'endog': self.yts, 'order': (2,1,2)}
        self.run_grid_search(estimator)

    def load_newh(self):
        '''
        Due to time-lapse in data collection and analysis, new data had been acquired that could be analyzed.
        Output: New dataframe containing completely unseen data.
        '''
        temp = pickle.load(open('pickle_pile/dfg.pkl', 'rb'))
        dfg = nstreams_filter(600, temp)
        dfg.set_index('date', inplace=True)
        dfg['year']=dfg.index.year
        return dfg

    def _find_holdout_date_thresh(self,channel):
        '''
        Input: channel
        Output: date range of holdout set
        '''
        self.cfilt(channel)
        self._make_holdout_split(self.cdf)
        return self.X_hold['date'].min(), self.cdf.index.max()

    def eval_holdout_data(self, channel):
        '''
        Used after adding freshly collected data.
        Evaluates models using previously gridsearch-optimized estimators. Currently, because of dummy variables, these need to be created early in the process to ensure proper dimensionality of categorical features. This step is not necessary if used in graphlab due its superious handling of categorical variables.
        Further, creation of dummie dictionary using training set and then adding dummy columns to holdout data encoded by dummy dictionary also works.
        '''
        dhmin, dhmax = self._find_holdout_date_thresh(channel)
        self.dfn = chan_filter(self.load_newh(), channel)
        self.dfn.sort_index(inplace=True)
        dft = self.dfn.query('date > @dhmax')
        self.dfu = pd.concat([self.cdf,dft])
        new_hold_num = self.dfu.query('date >= @dhmin').shape[0]
        lon = new_hold_num/7
        # dd = (self.X_hold.shape[0] + dft.shape[0])/7
        self._make_holdout_split(self.dfu, leaveout=lon)
        self._fchain_kfold_indicies()
        lassoCV_params = {'cv': self.fkfoldi,
                            'n_jobs':-1,
                            'alphas':np.logspace(-4,2,100)}
        ridgeCV_params =  {'cv': self.fkfoldi,
                            'alphas':np.logspace(-4,2,100),
                            'scoring':'mean_squared_error'}
        models = [RandomForestRegressor(**self.cvparams[channel]['RandomForestRegressor']['params']),
        GradientBoostingRegressor(**self.cvparams[channel]['GradientBoostingRegressor']['params']),
        LassoCV(**lassoCV_params),
        RidgeCV(**ridgeCV_params)]
        xh = self.X_hold.drop('date', axis=1)
        yh = self.y_hold.drop('date', axis=1).values
        plt.figure(figsize=(16,10))
        plt.plot(yh, label='True Values', color='black')
        for mod in models:
            mod.fit(self.Xtrn, self.ytrn)
            mod_name = mod.__class__.__name__
            ypred  = mod.predict(xh)
            mses = mean_squared_error(yh, ypred)
            rmse = np.sqrt(mses).mean()
            plt.plot(ypred, label = 'rmse: {0}, for model: {1}'.format(rmse, mod_name))

        plt.legend(loc='best')
        plt.show(block=False)

    def SMOrc_findbest(self):
        '''
        Output: pickled dictionary containing optimized model parameters for a specific channel.
        Runs gridsearch for optimal parameters for each model, for each channel. This is not ideal, but currently best option due to dramatic differences between individual channels.
        Further work involves creation/optimization of general model that will not be tailored for a specific channel.

        Fun Fact: Variable name comes from twitch.tv emote that depicts a brutish orc, representing the brute force method of optimization used.
        '''
        self.prep_arima()
        #Has to be done with dummie info so won't get error when instatiating in list with no input arguments
        models = [RandomForestRegressor(), GradientBoostingRegressor(), ARIMA(**self.arima_params)]
        channels = ['lirik', 'summit1g', 'imaqtpie', 'nl_kripp', 'destiny', 'admiral_bahroo']
        self.cvscores = {}
        for channel in channels:
            self.cvscores[channel] = {}
            for model in models:
                print 'Running ',model.__class__.__name__, ' for channel: ', channel
                self.kappa_search(channel, model)
                self.cvscores[channel][model.__class__.__name__] = {}
                if model.__class__.__name__ != 'ARIMA':
                    self.cvscores[channel][model.__class__.__name__]['params'] = self.gridsearch.best_params_
                    # self.mod = model(**self.gridsearch.best_params_)
                    mod = self.gridsearch.best_estimator_
                    self.cvscores[channel][model.__class__.__name__]['scores'] = cross_val_score(estimator=mod, X=self.Xtrn, y=self.ytrn, scoring='mean_squared_error', cv=self.fkfoldi, n_jobs=-1)
                else:
                    pass
                    # self.prep_arima(channel)
                    # mod = model(**self.arima_params)
                    # cvscore[channel][model.__class__.__name__]['scores'] = cross_val_score(estimator=mod, X=self.xts, y=self.yts, scoring='mean_squared_error', cv=self.ffkoldi, n_jobs=-1)
        pickle.dump(self.cvscores, open('pickle_pile/cross_val_SMOrc.pkl', 'wb'))


        # for estimator in models:
        #     self.cvscores[estimator] = {}
        #     for channel in channels:
        #         print 'Running ',estimator.__class__.__name__, ' for channel: ', channel
        #         self.kappa_search(channel, estimator)
        #         self.cvscores[estimator][channel] = {}
        #         if estimator.__class__.__name__ != 'ARIMA':
        #             self.cvscores[estimator][channel][params] = self.gridsearch.best_params_
        #             self.cvscores[estimator][channel][scores] = cross_val_score(estimator(**self.gridsearch.best_params_),self.Xtrn, self.ytrn, scoring='mean_squared_error', cv=self.fkfoldi)
        #         else:
        #             cvscore[estimator][channel][scores] = cross_val_score(estimator(**arima_params), scoring='mean_squared_error', cv=self.fkoldi)
        # pickle.dump(self.cvscores, open('pickle_pile/cross_val_SMOrc.pkl', wb))

    # def DansGame(self, channel):
    #
    #     self.cfilt(channel)
    #     self._make_holdout_split(self.cdf)
    #     self._fchain_kfold_indicies()
    #
    #     self.rfr = RandomForestRegressor(**self.cvparams[channel][RandomForestRegressor])
    #
    #     pass

    def run_arima(self):#use current build
        '''
        DEPRECATED:
        Primarily used for testing/debugging.
        Runs statsmodels ARIMA.
        '''

        self.xts = self.X_train.set_index('date')
        self.yts = self.y_train.set_index('date')
        self.yts.astype('float', inplace=True)
        self.arimod = ARIMA(endog = self.yts, order = (2,1,2))#, exog=self.xts)
        self.aresults = self.arimod.fit()

    def prep_arima(self, channel='lirik'):#use current build
        '''
        DEPRECATED:
        Required to 'prep' due to statsmodels' ARIMA not following the same flow as primarily used sklearn models.
        Abandoned in-lieu of R arima methods.
        '''
        self.cfilt(channel)
        self._make_holdout_split(self.cdf)
        self._fchain_kfold_indicies()
        self.xts = self.X_train.set_index('date')
        self.yts = self.y_train.set_index('date')
        self.yts.astype('float', inplace=True)
        self.arima_params = {'endog': self.yts, 'order': (2,1,2)}
        # self.arimod = ARIMA(endog = self.yts, order = (2,1,2))#, exog=self.xts)
        # self.aresults = self.arimod.fit()
    #
    # def EleGiggle(self, tname, dname, ivars):
        # self.r = robjects.r("""
        #     tset = read.csv("{0}")
        #     dset = read.csv("{1}")
        #     y = dset["AVG CCV's"]
        #     features = {2}
        #     X = train_set[features]
        #     X_test = test_set[features]
        #     fit = auto.arima(y, xreg=X)
        #     ypred = forecast(fit, xreg=X_test)
        # """.format(train_name, test_name, ivars))
    #     rp = robjects.r("""ypred['mean']""")[0]
    #     ypred = [rp[i] for i in range(len(rp))]
    #     return ypred


    def SeemsGood(self):
        '''
        Input:
        Output: Saves figures comparing model performance for each channel listed below.
        '''
        channels = ['lirik', 'nl_kripp', 'imaqtpie', 'summit1g']
        # channels = ['lirik']
        tscores = {}
        for channel in channels:
            tscores[channel] = {}
            self.cfilt(channel)
            self._make_holdout_split(self.cdf)
            self._fchain_kfold_indicies()
            # self.prep_arima(channel)
            # self.arimod = ARIMA(**self.arima_params)
            self.prep_arima(channel)
            lassoCV_params = {'cv': self.fkfoldi,
                                'n_jobs':-1,
                                'alphas':np.logspace(-4,2,100)}
            ridgeCV_params =  {'cv': self.fkfoldi,
                                'alphas':np.logspace(-4,2,100),
                                'scoring':'mean_squared_error'}
            lassy = LassoCV(**lassoCV_params)
            ridge = RidgeCV(**ridgeCV_params)
            lassy.fit(self.Xtrn,self.ytrn)
            ridge.fit(self.Xtrn, self.ytrn)
            x = {}
            yp = {}
            y = {}
            mses = {}
            # resA = []
            for j in xrange(len(self.kfolds)):
                Xtrain_indices, ytrain_indices = self.train_kfoldi[j]
                Xtest_indices, ytest_indices = self.test_kfoldi[j]
                xtrain = self.Xtrn.iloc[Xtrain_indices]
                ytrain = self.ytrn.iloc[ytrain_indices]
                xtest = self.Xtrn.iloc[Xtest_indices]
                ytest = self.ytrn.iloc[ytest_indices]
                # models = [RandomForestRegressor(**self.cvparams[channel]['RandomForestRegressor']['params'])]


                models = [RandomForestRegressor(**self.cvparams[channel]['RandomForestRegressor']['params']),
                GradientBoostingRegressor(**self.cvparams[channel]['GradientBoostingRegressor']['params']),Ridge(ridge.alpha_)]
                #Lasso(lassy.alpha_)]#, ]
                # models = [ARIMA(endog=yits, order= (2,1,2))] #exog=xits)]
                # models = ['ARIMA']
                for mod in models:
                    mod_name = mod.__class__.__name__
                    print '******'
                    print mod_name
                    print '******'
                    if j==0:
                        x[mod_name], y[mod_name], yp[mod_name], mses[mod_name] = [], [], [], []
                    if mod_name != 'ARIMA':
                        mod.fit(xtrain,ytrain)
                        ypred = mod.predict(xtest)
                        yp[mod_name].append(ypred)
                        y[mod_name].append(ytest.values)
                        x[mod_name].append(xtest)
                        mses[mod_name].append(mean_squared_error(ytest.values,ypred))
                    # elif mod=='ARIMA':
                    #     yits = self.y_train.iloc[ytrain_indices]
                    #     yits.set_index('date',inplace=True)
                    #     yits.astype('float', inplace=True)
                    #     xits = self.X_train.iloc[Xtrain_indices]
                    #     xits.set_index('date',inplace=True)
                    #     xits.astype('float', inplace=True)
                    #
                    #     ydts = self.y_train.iloc[ytest_indices]
                    #     ydts.set_index('date',inplace=True)
                    #     ydts.astype('float', inplace=True)
                    #     xdts = self.X_train.iloc[Xtest_indices]
                    #     xdts.set_index('date',inplace=True)
                    #     xdts.astype('float', inplace=True)
                    #     ivars = 'c({})'.format(u' '.join(list(xits.columns)).encode('utf-8')[1:-1])
                    #     tfold = pd.concat([xits,yits], axis=1)
                    #     tname = 'pickle_pile/tfold{0}.csv'.format(j)
                    #     # tname = 'tfold{0}.csv'.format(j)
                    #
                    #     tname = ''.join(tname).encode('utf-8')
                    #     tfold.to_csv(tname)
                    #
                    #     # tname = 'pickle_pile/tfold.csv'
                    #
                    #     dfold = pd.concat([xdts,ydts], axis=1)
                    #     dname = 'pickle_pile/dfold{0}.csv'.format(j)
                    #     # dname = 'pickle_pile/dfold.csv'
                    #     dfold = to_csv(dname)
                    #     try:
                    #         ypred = self.EleGiggle(tname, dname, ivars)
                    #         yp[mod_name].append(ypred)
                    #         y[mod_name].append(ytest.values)
                    #         x[mod_name].append(xtest)
                            # mses[mod_name].append(mean_squared_error(ytest.values,ypred))
                    #     except:
                    #         print 'Nope'

                    # else:
                    #     ares = mod.fit()
                    #     ypred= ares.predict(start=ydts.index.min(), end=ydts.index.max())
                    #     yp[mod_name].append(ypred)
                    #     y[mod_name].append(ytest.values)
                    #     x[mod_name].append(xtest)
                        # mses[mod_name].append(mean_squared_error(ytest.values,ypred))

            self.xym = {}
            fig = plt.figure(figsize=(16,10))
            plt.tick_params(labelsize=18)
            for mod in models:
                mod_name = mod.__class__.__name__
                self.xym[mod_name] = {}
                self.xym[mod_name]['y'] = list(itertools.chain.from_iterable(y[mod_name]))
                self.xym[mod_name]['x'] = list(itertools.chain.from_iterable(x[mod_name]))
                self.xym[mod_name]['yp'] = list(itertools.chain.from_iterable(yp[mod_name]))
                self.xym[mod_name]['rmse'] = np.average(np.sqrt(mses[mod_name]))
                if mod_name=='RandomForestRegressor':
                    plt.plot(self.xym[mod_name]['yp'], color='blue', linewidth=2, label=mod_name+' rmse: {0}'.format(self.xym[mod_name]['rmse']))
                elif mod_name=='GradientBoostingRegressor':
                    plt.plot(self.xym[mod_name]['yp'], 'g--', linewidth=2, label=mod_name+' rmse: {0}'.format(self.xym[mod_name]['rmse']))
                elif mod_name=='Ridge':
                    plt.plot(self.xym[mod_name]['yp'], 'r--', linewidth=2, label=mod_name+' rmse: {0}'.format(self.xym[mod_name]['rmse']))
            # plt.scatter(self.Xtrn.index[:len(self.xym[mod_name]['y'])], self.xym[mod_name]['y'], color='green', marker='o', label='true')
            plt.plot(self.xym[mod_name]['y'], color='magenta', label='true', linewidth=2)
            plt.legend(loc='best', prop={'size':14})

            # plt.show()
            plt.savefig('figures/cv_{0}_rflmodel.png'.format(channel))
            # fy = list(itertools.chain.from_iterable(forest_y))
            # fyp = list(itertools.chain.from_iterable(forest_yp))

            # tscores[channel]['forest'] = np.sqrt(msesf)

            # plt.plot(fy, color='blue', label='True Values')
            # plt.plot(fyp, color = 'green', label='Predicted')
            # plt.plot(x=forest_x, y=forest_y, color='blue', label='True Values')
            # plt.plot(x=forest_x, y=forest_yp, color = 'green', label='Predicted')
            # rolmean = pd.Series.rolling(forest_y, window=7).mean()
            # rolmeanp = pd.Series.rolling(forest_yp, window=7).mean()
            # plt.plot(rolmean, color='red', label='True rolling mean')
            # plt.plot(rolmeanp, color='black', label='predicted rolling mean')
            # plt.legend(loc='best')
            # rmse = tscores[channel]['forest'].mean()
            # plt.title('Cross Val predictions for {0} with rmse = {1}'.format(channel, rmse))
            # fig.savefig('CVFor_TT{0}.png'.format(channel))
            # plt.imsave('CVFor_{0}.png'.format(channel))


            # tscores[channel][ARIMA] = np.sqrt(msesA)

            # self.rmses, self.r2_scores = np.sqrt(mses), np.array(r2s)




    # def kappa_claus(self):
    #     self._make_holdout_split(self.df)
    #     self._fchain_kfold_indicies()
    #     rfr_grid_params =  {'max_features': 'sqrt', 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'mse', 'n_estimators': 300}
    #     self.rfrgs = RandomForestRegressor(**rfr_grid_params)
    #     self.rfrgs.fit(self.Xtrn, self.ytrn)
    #     # mses = []
    #     # r2s = []
    #     # #use cross_val_predict to get predictions for different sections
    #     # for j in xrange(len(self.kfolds)):
    #     #     print '******'
    #     #     print 'Evaluating Fold #{0}'.format(j)
    #     #     print '******'
    #     #     Xtrain_indices, ytrain_indices = self.train_kfoldi[j]
    #     #     Xtest_indices, ytest_indices = self.test_kfoldi[j]
    #     #     xtrain = self.Xtrn.iloc[Xtrain_indices]
    #     #     ytrain = self.ytrn.iloc[ytrain_indices]
    #     #     xtest = self.Xtrn.iloc[Xtest_indices]
    #     #     ytest = self.ytrn.iloc[ytest_indices]
    #     #     # self.rfrcv.fit(xtrain.values, ytrain.values)
    #     #     ypred = self.rfrgs.predict(xtest)
    #     #     mses.append(mean_squared_error(ytest.values, ypred))
    #     #     r2s.append(r2_score(ytest.values, ypred))
    #     # self.rmses, self.r2_scores = np.sqrt(mses), np.array(r2s)






if __name__ == '__main__':
    mk = Kappa('pickle_pile/dff.pkl')
    # mk.eval_holdout_data('lirik')
    mk.SeemsGood()
    # mk.SeemsGood()
    # mk.kappa_claus()
    # mk.prep_arima()
    # mk.SMOrc_findbest()

    # mk.cfilt('lirik')
    # mk.cdf.sort_index(inplace=True)
    # mk._make_holdout_split(mk.cdf)
    # mk._fchain_kfold_indicies()
    # mk.run_cvmod('lirik')
    # mk.test_stationarity('lirik')
    # mk.golden_kappa()
