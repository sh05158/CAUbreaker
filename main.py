from random import randrange
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.remote.webelement import WebElement



target = 'https://sugang.cau.ac.kr/'

driver = webdriver.Chrome(executable_path='./chromedriver')
driver.get(url=target)

driver.implicitly_wait(5)

time.sleep(1)

ID = 'sh05158'
PW = 'songfish320'
SUBJECT = '데이타베이스'


driver.switch_to.frame('Main')

# time.sleep(100)
driver.find_element(By.NAME,'userID').send_keys(ID)
time.sleep(1)

driver.find_element(By.NAME,'password').send_keys(PW)
time.sleep(1)

driver.find_element(By.ID,'btn-login').click()
time.sleep(1)
driver.implicitly_wait(5)

time.sleep(1)

driver.switch_to.default_content()

driver.switch_to.frame('Main')
driver.switch_to.frame('coreMain')

driver.find_element(By.ID,'menu_sugang').click()

time.sleep(1)

driver.find_element(By.ID,'menu03').click()

time.sleep(1)

driver.find_element(By.ID,'pKorNm').send_keys(SUBJECT)

select = Select(driver.find_element(By.ID,'pDay'))
select.select_by_index(2)

time.sleep(1)

while True:
    
    tempVal = 0
    temp2 = None
    
    while tempVal < 1:
        driver.find_element(By.ID,'btnSearch').click()
        
        element = WebDriverWait(driver, 10).until(EC.invisibility_of_element_located((By.ID, 'load_gridLecture')))
        
        time.sleep(1)

        temp = driver.find_element(By.ID,'gridLecture')

        temp = temp.find_elements(By.TAG_NAME,'tbody')[0]
        
        temp2 = temp.find_element(By.ID,'1')
        
        temp = temp2.find_elements(By.TAG_NAME,'td')[12]
        
        tempVal = int(str(temp.get_attribute('innerHTML')).split('<br>')[0])
        
        print('tempVal' + str(tempVal))
        
    temp2.click()
    
    time.sleep(1)
    
    try:
        driver.find_element(By.CLASS_NAME,'jconfirm-closeIcon').click()
        time.sleep(2)
    except:
        print('Success')
        break
        
    
    

    