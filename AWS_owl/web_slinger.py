from selenium import webdriver
from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
import selenium.webdriver.support.ui as ui
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import json
import os
import requests
import time

class web_slinger(object):
    '''
    Has methods to navigate the gamoloco webiste.
    Allows login/logout, entering email/password, etc

    Fun Fact: Spidey is a web slinger.
    '''
    def __init__(self):
        '''
        Input:
        Output: Instantiates and defines xpath for many common locations of interaction through selenium.
        '''

        self.url_base = 'https://gamoloco.com/'
        self.login_button_xp = '''//a[@ng-click="$root.$emit('login.trigger')"]'''
        self.login_email_field_xp = '//input[@type="email"]'
        self.login_pw_field_xp = '//input[@type="password"]'
        self.login_submitter_xp = '//button[@type="submit"]'
        self.export_table_button_xp = '//button[@ng-click="exptctrl.export()"]'
        self.data_path = os.getcwd()+'/data'
        self.logout_button_xp = '//a[@ng-click="$root.$user.logout()"]'
        self.dlpath = '/home/derek/Downloads/'
        self.logged_in = False
        self.profile_button_xp = '//a[@ui-sref-active="active"]'


    def _create_fp(self):
        '''
        Input:
        Output: Sets up selenium firefox profile for scraping.

        Important purposes:
        Making firefox not ask if I am sure I want to download which would require and additional click.
        Setting file download destination.
        '''
        self.profile = webdriver.FirefoxProfile()
        self.profile.set_preference('browser.download.folderList',2)
        self.profile.set_preference('browser.download.manager.showWhenStarting', False)
        self.profile.set_preference('broser.download.dir', self.data_path)
        self.profile.set_preference('browser.helperApps.neverAsk.saveToDisk','application/xlsx;text/csv;application/octet-stream')

    def _splat_on_web(self):
        '''
        Input:
        Output: Creates webdriver with firefox profile and sends to base url.
        '''
        self.driver = webdriver.Firefox(firefox_profile=self.profile)
        self.driver.implicitly_wait(4)
        self.driver.get(self.url_base)

    def _click_login_button(self):
        '''
        Clicks button which drops down email/password dialogue for login.
        '''
        self.link = self.driver.find_element_by_xpath(self.login_button_xp)
        self.link.click()

    def _send_login_email(self):
        '''
        Uses system variables
        '''
        self.link = ui.WebDriverWait(self.driver, 3).until(EC.presence_of_element_located(By.XPATH, self.login_email_field_xp))
        self.link.send_keys(os.environ.get('gamoloco_email'))

    def _send_login_pw(self):
        '''
        My password is:
        located in bashprofile.
        '''
        self.link = self.driver.find_element_by_xpath(self.login_pw_field_xp)
        self.link.send_keys(os.environ.get('gamoloco_password')+Keys.RETURN)

    def _click_login_submitter(self):
        '''
        clicks additional button in login drop-down after email and password are entered.
        '''
        self.link = self.driver.find_element_by_xpath(self.login_submitter_xp)
        self.link.click()

    def web_crawl(self,target_url):
        '''
        Input: target page's url.
        Output: Spidey crawls on his web to target url.
        '''
        self.driver.get(target_url)

    def extract_data(self):
        '''
        Clicks download data table button. Button needs to be visible requiring pyvirtualdisplay on AWS.
        Often the last thing to load on the page requiring a webdriverwait.
        to be called on data_url.
        '''
        #try using waits
        self.link = ui.WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located(
                    (By.XPATH, self.export_table_button_xp)
                    )
                )
        # self.link=self.driver.find_element_by_xpath(self.export_table_button_xp)
        self.link.click()

    def logout_clicker(self):
        '''
        Clicks logout button
        '''
        self.link = self.driver.find_element_by_xpath(self.logout_button_xp)
        self.link.click()

    def spin_web(self):
        '''
        Used to start process:
        1. Create firefox profile, 2. instatiates webdriver and sends to base_url
        '''
        self._create_fp()
        self._splat_on_web()

    def _send_loginfo(self):
        '''
        DEPRECATED:
        Use individual functions within selenium actionchain to ensure proper ordering of many events.
        '''
        self._send_login_email()
        self._send_login_pw()


    def click(self):
        '''
        Who doens't like clicking stuff?
        '''
        self.link.click()



    def cookie_logger(self):
        '''
        Most consistent way of determining login status was found to be counting the number of cookies on the webdriver.
        Used instead of status-specific button like "login" or "logout" because the buttons are always present albeit hidden and unclickable. Additionally, cookie checking is faster.
        '''
        if len(self.driver.get_cookies()) > 3:
            self.logged_in = True
            print 'Tastes like logged in'
        else:
            self.logged_in = False
            print 'Tastes bad, like Im logged out'

    def login(self):
        '''
        Logins in to gamoloco.com
        Utilizes action chains to ensure proper ordering and timing of login events.
        '''
        if self.logged_in == False:
            try:
                lb = self.driver.find_element_by_xpath(self.login_button_xp)
                emf = self.driver.find_element_by_xpath(self.login_email_field_xp)
                pwf = self.driver.find_element_by_xpath(self.login_pw_field_xp)
                self.login_actions = ActionChains(self.driver)
                self.login_actions.click(lb)
                self.login_actions.send_keys_to_element(emf,
                                                os.environ.get('gamoloco_email'))
                self.login_actions.send_keys_to_element(pwf,
                                                os.environ.get('gamoloco_password'))
                self.login_actions.send_keys(Keys.RETURN)
                self.login_actions.perform()

            except:
                print 'action chain failed'
            try:
                self._click_login_submitter()
                print 'clicked submitter again'
            except:
                print 'could not click submitter again'

            # time.sleep(1)
            # self.logged_in_check()

            #
            #
            # print 'Not currently logged in. Attmpting to login'
            # self._click_login_button()
            # try:
            #     self._send_login_email()
            #     self._send_login_pw()
            #     try:
            #         self._click_login_submitter()
            #         print 'pushed submitter button again'
            #     except:
            #         print 'could not press submit button again'
            #     finally:
            #         self.logged_in_check()
            #
            # except:
            #     print 'error in sending loginfo'
            #     print 'try again'
            #     self.logged_in_check()

        else:
            print 'already logged in'







    def login_to_web(self):
        '''
        DEPRECATED:
        Does not use selenium action chains to ensure proper odering/timing.
        This function was very helpful in debugging login process.
        Use >> spidey.login() instead.

        '''
        if self.logged_in == False:
            print 'Not currently logged in. Attempting to login'
            self._click_login_button()
            try:
                self._send_loginfo()
                print 'initial send success'
            except:
                self._click_login_button()
                try:
                    self._send_loginfo()
                    print 'had to click button to dropdown'
                except:
                    print 'failed to send loginfo'
            try:
                self._click_login_submitter()
                print 'attempting first login submit'
                #Sometimes need to click again to successfully login
                try:
                    self._click_login_submitter()
                    print 'had to click login submitter button again'
                except:
                    print 'login submit occurred correctly 1st time'

            except:
                print 'login submitter not present, start over'


        else:
            print 'Already logged in'

    def logged_in_check(self):
        '''
        DEPRECATED:
        Uses inefficient and confounding method for testing login status. Login button always visible to webdriver, although cannot be seen on page due to ng-hide methods within html code.
        Current working version uses cookies. This has the added benefit of making Spidey happy.
        '''
        #Checking using # of cookies another, 3 not logged in, 5 logged in
        try:
            self._click_login_button
            self.logged_in = False
            print 'Logged in?: ', self.logged_in
        except:
            self.logged_in = True
            print 'Logged_in?: ', self.logged_in



if __name__ =='__main__':
    spidey = web_slinger()
    spidey.spin_web()
    spidey.login_to_web()
    day = '17'
    month = '7'
    year = '2016'
    data_url = 'https://gamoloco.com/scoreboard/channels/daily/{0}/{1}/{2}'.format(day,month,year)
    spidey.web_crawl(data_url)
