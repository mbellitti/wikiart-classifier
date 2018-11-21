import pickle
from image_scraper import *
import pandas as pd
from multiprocessing import Pool

N_rows = 100

filename_url = 'artworks_urls_full.pkl' # pickled file with all artwork urls

with open('../data/'+filename_url, 'rb') as f:
    url_list = pickle.load(f)

url_list = url_list[:N_rows]

pool = Pool(processes=4)
list_data = pool.map(get_meta_data, url_list)

pd.DataFrame(list_data).to_csv("database.csv",index=False)
