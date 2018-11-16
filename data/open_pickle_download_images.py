import pickle
import json
import requests
from bs4 import BeautifulSoup
import sys
sys.path.insert(0, '../src')
from image_scraper import *

filename1 = 'artworks_urls_full.pkl' 
f1 = open(filename1, 'rb')
url_list=pickle.load(f1)
f1.close()

print('Total number of urls', len(url_list))


for i in range(1):
    meta_data_i=get_meta_data(url_list[i])
    print(meta_data_i)
    

#ToDo: before dowloading images, check if total memory is bigger than 1 GB.
for i in range(10):
    image_url_i=image_html_fn(url_list[i])
    print(image_url_i)
    image_save_as_file_fn(image_url_i)
    #print(meta_data_i)    