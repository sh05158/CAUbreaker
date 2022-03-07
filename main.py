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
import platform



target = 'https://sugang.cau.ac.kr/'

if platform.system() == 'Darwin':
    driver = webdriver.Chrome(executable_path='./chromedriver')
elif platform.system() == 'Windows':
    driver = webdriver.Chrome(executable_path='./chromedriver.exe')
else:
    print('unknown OS Error')
            
            
driver.get(url=target)

driver.implicitly_wait(5)

time.sleep(1)

ID = ''
PW = ''
SUBJECT = '데이타베이스'
NUM = 2  #1 => 월, 2 => 화, 3=> 수, 4=>목, 5=>금..


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
select.select_by_index(NUM)

time.sleep(1)

while True:
    
    tempVal = -1

    temp2 = None
    
    while tempVal < 1:
        def refresh():
            global temp2
            
            driver.find_element(By.ID,'btnSearch').click()
            
            element = WebDriverWait(driver, 10).until(EC.invisibility_of_element_located((By.ID, 'load_gridLecture')))
            
            time.sleep( randrange(0,50)*0.01 + 0.5)

            temp = driver.find_element(By.ID,'gridLecture')

            temp = temp.find_elements(By.TAG_NAME,'tbody')[0]
            
            temp2 = temp.find_element(By.ID,'1')
            
            temp = temp2.find_elements(By.TAG_NAME,'td')[12]
            
            tempVal = int(str(temp.get_attribute('innerHTML')).split('<br>')[0])
            
            print('tempVal' + str(tempVal))
            
            return tempVal
        
        try:
            tempVal = refresh()
            try:
                tempVal = int(tempVal)
            except:
                tempVal = 0
                
        except Exception as e:
            print('Error : '+str(e))
            
        
            
    def attemptOK(temp2):
        temp2.find_elements(By.TAG_NAME,'td')[0].click()
        
        time.sleep(1)
        
        # driver.find_element(By.CLASS_NAME,'jconfirm-closeIcon').click()
        elem = driver.find_elements(By.CLASS_NAME,'jconfirm-open')

        if len(elem) > 0:
            if '정상적으로' in str( elem[0].find_elements(By.TAG_NAME,'p')[0].get_attribute('innerHTML') ):
                print('Success')
                driver.find_element(By.CLASS_NAME,'jconfirm-closeIcon').click()
                time.sleep(0.3)
                return True
            else:
                print('Fail')
                driver.find_element(By.CLASS_NAME,'jconfirm-closeIcon').click()
                time.sleep(0.3)
                
                return False
        return False        
        
    try:
        if attemptOK(temp2):
            break
    except Exception as e:
        print('Error2 : '+str(e))
            
    
    

        
    
    

    