import pickle
import sys
from tqdm import tqdm
import pandas as pd
from image_scraper import *
from multiprocessing import Pool
import time

def job(url):
    """Fetches metadata and catches possible exceptions

    Arguments:
        url: url to be fetched

    Returns:
        sucessful fetching: a list with the meta data as its first entry and
            None as its second
        Connection Error: a list with None as its first entry and the url as
            its second
        Type Error (can occurr if metadata isn't found/correctly read on page):
            [None, None]
            additionaly writes url to extra file."""

    from urllib.parse import urlparse
    try:
        if urlparse(url).netloc.split('.')[-1] != 'org':
            raise TypeError("Nonvalid url: top level domain is not '.org': {}".format(url))
        return [get_meta_data(url), None]
    except requests.exceptions.RequestException as e:
        # print('############Connection Error#########')
        return [None, url]
    except TypeError:
        filename = 'problematic_urls.txt'
        f = open(filename, 'a')
        f.write(url + '\n')
        f.close()
        return [None, None]

def save_csv(file_name, N_rows=None, N_rows_start = 0, file_url_list='artworks_urls_full.pkl', processes = 8):
    """Downloads all the metadata in the pickled url list and saves to csv.

    Arguments:
        file_name: name of the csv file in which data would be saved
        N_rows: number of urls to download, by default downloads all of them
        url_list: picked file. Each line is one that will be downloaded

    Returns:
        None

    Examples:
        For N_rows=2, this function saves the file with following entry
        ,_id,title,artistname,image,year,style,genre,file_name
        0,5772847bedc2cb3880fded05,self-portrait,hans von aachen,https://uploads4.wikiart.org/images/hans-von-aachen/self-portrait-1574.jpg,1574,mannerism (late renaissance),self-portrait,self-portrait-1574.jpg!Large.jpg
        1,5772847bedc2cb3880fded75,two laughing men (double self-portrait),hans von aachen,https://uploads1.wikiart.org/images/hans-von-aachen/two-laughing-men-double-self-portrait-1574.jpg,1574,mannerism (late renaissance),self-portrait,two-laughing-men-double-self-portrait-1574.jpg!Large.jpg

    """

    with open('../data/'+file_url_list, 'rb') as f:
        if N_rows is None:
            url_list = pickle.load(f)
        else:
            url_list = pickle.load(f)[N_rows_start:N_rows]
    pool = Pool(processes=processes)

    start = time.time()

    #header for problematic_urls file
    filename = 'problematic_urls.txt'
    f = open(filename, 'a')
    f.write('run on ' +  time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime()) +':\n')
    f.close()

    # go through url list
    data = pool.map(job, url_list)
    list_data = [i[0] for i in data if i[0] != None]
    missed_urls = [i[1] for i in data if i[1] != None]
    # print("Successful:",len(list_data))
    # print("Errors: ",len(missed_urls))
    del data

    # redo all missed urls until no one is left behind
    while len(missed_urls) >0:
        # print('another round')
        data = pool.map(job, missed_urls)
        list_data_add = [i[0] for i in data if i[0] != None]
        missed_urls = [i[1] for i in data if i[1] != None]
        del data
        list_data.extend(list_data_add)
        del list_data_add
        # print("Successful:",len(list_data))
        # print("Errors: ",len(missed_urls))

    end = time.time()
    pd.DataFrame(list_data).to_csv(file_name,index=False)

    pool.close()

    return end-start

if __name__ == '__main__':
    chunk_size = 100
    file_url_list='artworks_urls_full.pkl'
    len_url_list = -1                  # -1 for full run, otherwise test length
    chunk_to_start_with = 1            # set to 1 for full run

    if len(sys.argv) > 0:
        chunk_to_start_with = int(sys.argv[1])

    if len_url_list == -1:
        with open('../data/'+file_url_list, 'rb') as f:
            url_list = pickle.load(f)
            len_url_list = len(url_list)
            del url_list

    chunk_borders = list(range(0,len_url_list, chunk_size))
    chunk_borders.append(len_url_list)
    num_chunks = len(chunk_borders) -1

    # loop through chunks
    start = time.time()
    for chunk_id in tqdm(range(chunk_to_start_with -1, num_chunks)):
        run_time = save_csv(
            "database/database{0}of{1}.csv".format(chunk_id +1, num_chunks),
            N_rows_start = chunk_borders[chunk_id], N_rows = chunk_borders[chunk_id +1],processes=4)
    end = time.time()
    total_time = end - start

    print('downloading time: {:.1f}s'.format(total_time))
