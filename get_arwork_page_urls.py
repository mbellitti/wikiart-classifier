
import requests
from bs4 import BeautifulSoup
import pickle

def get_imagepage_urls(test = False):
    """For every artwork there is a separate page in wikiart. This function
    gets the urls of all these pages.
    
    Input:
        test: if True, then only the artwork urls of a single artist
        are returned
    
    Output: 
        artworks_urls: list of urls of all (execpt for test == True) artwork
        pages 
        
        skipped_urls: if request.get(url) fails for a given artist page, 
        the this url is skipped and written into this list. 
        A successful run returns an empty skipped_urls list. 
    
    The script relies on the fact, that there are pages listing all artists
    with the same initial letter with urls of the Form
    https://www.wikiart.org/en/alphabet/<letter>/text-list
    and for every artist there is a page listing all his or her artworks 
    with an url of the form
    https://www.wikiart.org/en/<artist>/all-works/text-list .
    Wikiart has one artist listed starting with a number instead of a letter.
    The respective page takes %F8 (Ø) at the <letter> place.
    """
    
    if test ==True:
        end = 1
    else:
        end = None
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    numericals = '%F8'                          
    alphnum = [i for i in alphabet] + [numericals]
    
    base_url_str = 'https://www.wikiart.org/en/alphabet/X/text-list'
    
    
    # get list of all all-works urls
    ## get list of all artist-urls
    artist_urls = []  
    for letter in alphnum[:end]:
        url = base_url_str.replace('X', letter)
        print(url)
        cur_artist_urls = get_textlist_urls(url)
        artist_urls.extend(cur_artist_urls)
    ## change to urls with list of works
    artistworks_urls = []
    for url in artist_urls:
        artistworks_urls.append(url + '/all-works/text-list')
    
    # get list of all artwork urls
    artworks_urls = []
    skipped_urls = []
    for url in artistworks_urls[:end]:
        print(url)
        try:
            cur_artworks_urls = get_textlist_urls(url)
        #hopefully handling connection errors -- couldn't test that, yet.
        except requests.exceptions.RequestException as e: 
            print(e)
            print('url: {} was skipped'.format(url))
            skipped_urls.append(url)
        artworks_urls.extend(cur_artworks_urls)
    
    return artworks_urls, skipped_urls

def get_textlist_urls(url):
    """Takes an url of an text-list page on wikiart and returns a list with 
    the urls in the list. 
    
    Input:
        url: url of an text-list page on wikiart
    
    Output:
        urls: a list of the urls of all links in the text-list
    
    Examples of text-list pages on wikiart::
        *   https://www.wikiart.org/en/alphabet/<letter>/text-list
        *   https://www.wikiart.org/en/<artist>/all-works/text-list
    """
    
    #host = 'www.wikiart.org' in our case
    from urllib.parse import urlparse
    host = urlparse(url).netloc
    scheme = urlparse(url).scheme
    
    # get soup
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # The list of artist on the site page is in the 'main' tag in the 'ul' tag.
    # The links are in 'a' tags.
    links = soup.main.ul.find_all('a')
    
    urls = []
    for link in links:
        urls.append(scheme + '://' + host + link.get('href'))
    
    return urls


def main():
    
    artworks_urls, skipped_urls = get_imagepage_urls(test = False)
    
    # save urls
    filename1 = 'artworks_urls.pkl'
    f1 = open(filename1, 'wb')
    pickle.dump(artworks_urls, f1)
    f1.close()
    # save skipped urls
    filename2 = 'skipped_urls.pkl'
    f2 = open(filename2, 'wb')
    pickle.dump(skipped_urls, f2)
    f2.close()
    
    print(len(artworks_urls))
    
if __name__ == '__main__':
    main()
