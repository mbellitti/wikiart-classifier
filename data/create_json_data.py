import pickle
import json
import requests
from bs4 import BeautifulSoup
import sys
sys.path.insert(0, '../src')
from image_scraper import *
import pandas as pd
import numpy as np

    
def save_json(file_name, N_rows=2):
    """
    Description:
        It opens the pickle-file of urls and saves a JSON data file with meta data and file-name as the last column. 
    input:
        file_name: name of the JSON file in which data would be saved

    output:
        None

    Example:
        For N_rows=2, this function saves the file with following entry
         {"_id":{"0":"5772847bedc2cb3880fded05","1":"5772847bedc2cb3880fded75"},
            "title":{"0":"self-portrait","1":"two laughing men (double self-portrait)"},
            "artistname":{"0":"hans von aachen","1":"hans von aachen"},
            "image":{"0":"https:\/\/uploads4.wikiart.org\/images\/hans-von-aachen\/self-portrait-1574.jpg","1":"https:\/\/uploads1.wikiart.org\/images\/hans-von-aachen\/two-laughing-men-double-self-portrait-1574.jpg"}
            ,"year":{"0":"1574","1":"1574"},"style":{"0":"mannerism (late renaissance)","1":"mannerism (late renaissance)"},
            "genre":{"0":"self-portrait","1":"self-portrait"},
            "file_name":{"0":"self-portrait-1574.jpg!Large.jpg","1":"two-laughing-men-double-self-portrait-1574.jpg!Large.jpg"}}
    To-do:
        1.It should give ignore the urls which give error and move on to the other urls in the list.
        2. N_rows should be len(url_list). As of now, the function gives an error for N_rows>166. 
    
    
    
    """
    filename_url = 'artworks_urls_full.pkl' 
    f1 = open(filename_url, 'rb')
    url_list=pickle.load(f1)
    f1.close()
    f2 = open(file_name, 'w')
    df = pd.DataFrame(index=np.arange(0, N_rows), columns=('_id','title','artistname','image', 'year', 'style', 'genre', 'file_name') )

    for i in range(N_rows):
        meta_data_i=get_meta_data(url_list[i])
        image_url_i=image_html_fn(url_list[i])
        meta_data_i.update({'file_name':image_url_i.split('/')[-1]})
        df_temp=pd.DataFrame(meta_data_i, index=[0])
        df.loc[i]=df_temp.iloc[0]

    df.to_json(f2)
    return None


f2='artwork_data.json'    

save_json(f2)    
#i=166 #bad url.
