# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 08:15:47 2022

@author: MYSTERIOUS
"""

import bs4
import requests
import sys
import time
from bs4 import BeautifulSoup

try:
    page=requests.get('https://www.cricbuzz.com/')
except Exception as e:
    error_obj,error_type,error_info=sys.exc_info()
    time.sleep(2)
    
    

print(page)  
soup=BeautifulSoup(page.text,'html.parser')
                                     
print(soup) 
link=soup.find_all('div',attrs={'class': "cb-nws-intr" })
print(link)
for i in link:
    print(i.text)
    print('\n')