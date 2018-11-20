import pickle
import json
import requests
from bs4 import BeautifulSoup
import sys
#sys.path.insert(0, '../src')
from image_scraper import *
import pandas as pd
import numpy as np

    
def save_csv(file_name, N_rows=2):
    """
    Description:
        It opens the pickle-file of urls from "data" folder and saves a CSV file in "data" folder with meta data and file-name as the last column. 
    input:
        file_name: name of the csv file in which data would be saved

    output:
        None

    Example:
        For N_rows=2, this function saves the file with following entry
        ,_id,title,artistname,image,year,style,genre,file_name
        0,5772847bedc2cb3880fded05,self-portrait,hans von aachen,https://uploads4.wikiart.org/images/hans-von-aachen/self-portrait-1574.jpg,1574,mannerism (late renaissance),self-portrait,self-portrait-1574.jpg!Large.jpg
        1,5772847bedc2cb3880fded75,two laughing men (double self-portrait),hans von aachen,https://uploads1.wikiart.org/images/hans-von-aachen/two-laughing-men-double-self-portrait-1574.jpg,1574,mannerism (late renaissance),self-portrait,two-laughing-men-double-self-portrait-1574.jpg!Large.jpg

    
    To-do:
        1. It should ignore the urls which give error and move on to the other urls in the list.
        2. N_rows should be len(url_list).
    
    """
    filename_url = 'artworks_urls_full.pkl' 
    f1 = open('../data/'+filename_url, 'rb')
    url_list=pickle.load(f1)
    f1.close()
    
    df = pd.DataFrame(index=np.arange(0, N_rows), columns=('_id','title','artistname','image', 'year', 'style', 'genre', 'file_name') )

    for i in range(N_rows):
        meta_data_i=get_meta_data(url_list[i])
        image_url_i=image_html_fn(url_list[i])
        meta_data_i.update({'file_name':image_url_i.split('/')[-1]})
        df_temp=pd.DataFrame(meta_data_i, index=[0])
        df.loc[i]=df_temp.iloc[0]

    #df.to_json(f2)
    df.to_csv('../data/'+file_name)
    return None

save_csv('pd_data_frame.csv')