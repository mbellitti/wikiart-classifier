import requests
from bs4 import BeautifulSoup

# finds the  html address
# TODO find better programm structure to include it in crawler
# idea: class that takes a url and loads the page. -> methods to get jpg and metadata

def image_html_fn(url):
    """
    input: url of Wikiart page, which has a single image
    output: web url of the source image.
    """

    page_trial_wikiart = requests.get(url)

    page_trial_wikiart_html = BeautifulSoup(page_trial_wikiart.content, 'html.parser')

    search_image_line=page_trial_wikiart_html.find_all(class_='ms-zoom-cursor')

    assert len(search_image_line) != 0  #make sure the search was successful

    str_list=str(search_image_line[0])

    # used this class because all (?) the webpages seem to have a property
    # that when you take your mouse it zooms in.
    # This is how I am trying to find source image
    # print(search_image_line)

    # I replaced that piece in favor of shorter version below.
    # I think, they are equivlent under the assumption that there is only one
    # picture on the site.
    """
    temp_var=[]
    for i in range(len(str_list)):
    ##searches for the word "src" and then copies the address after the word "src"
        j=0
        #if str_list[i:i+1]
        if str_list[i:i+3]=="src":
            start_index=str_list.find('\"',i+3)
            #print(start_index)
            end_index=str_list.find('\"', start_index+1)
            #print(end_index)
            for j in range(start_index+1, end_index):
                temp_var.append(str_list[j])
            #print(temp_var)
    return ''.join(temp_var)
    """

    marker_idx = str_list.find('src')           # finds first instance of 'src'
    assert marker_idx != -1, "couldn' find 'src' in string supposed to contain url"
    beg = str_list.find('\"', marker_idx) + 1   # find beginning of url string
    end = str_list.find('\"', beg)              # find end of url string
    print(str_list[beg:end])
    return str_list[beg:end]

def get_meta_data(url):
    """Takes an url of an 'artwork' page in wikiart and returns the meta data
    as a dictionary.

    As Metadata are considered: The url, the title, the artist and all
    additional infomation in the column to the right of the picture.
    The keys are: 'url', 'title', 'artist' and the actual names of the
    categories on the website.
    Thus the keys depend on the categories given on the website.
    """
    # observation1: meta data is found in 'article' tag
    # observation2: the metadata categories are in 's' tag.
    #       --> Not all categories for all artworks!!
    # -> is the article tag used exclusively for this on the site?

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    article = soup.find_all('article')
    assert len(article) == 1, "Found more than one 'article' tag on page"
    article = article[0]

    # get metadata categories
    catgs = article.find_all('s')

    # Get actual text strings and getting rid of colon at the end of category.
    for i in range(len(catgs)):
        catgs[i] = str(catgs[i].string)[:-1]
    num_catgs = len(catgs)
    #print(catgs)

    # Get rid of stuff that causes us troubles (i.e. pure text we don't need):
    #   *   <script> tag
    #   *   'order reproduction' box
    # If you find more exception that need to be handled this is a good place
    # to integrate them into the code.
    if article.script != None:
        article.script.extract()
    if article.find_next(class_ = 'order-reproduction')!= None:
        article.find_next(class_ = 'order-reproduction').extract()

    # Retrieve remaining strings
    content_list = list(article.stripped_strings)
    # print(content_list)

    # Create dictionary for metadata from this list using categories.
    # The first two lines of text left are title and author.
    #   --> always true??
    meta_dict = {'url': url,
        'title' : content_list[0], 'artist' : content_list[1]}

    # Get content list as a single string
    #
    # last category is 'Share'; we don't want to share
    content_str = ''.join(content_list)
    for catg_idx in range(len(catgs)-1) :
        beg = content_str.find(catgs[catg_idx]) + len(catgs[catg_idx]) +1
        end = content_str.find(catgs[catg_idx + 1])
        meta_dict.update({catgs[catg_idx]: content_str[beg:end]})

    # we can't use this simpler method that would not require extracting the
    # categories, because entrys may have several lines in html
    # e.g. 'c. 1504' has the c. separated
    """
    while i+1 <= len(content_list)-1:    #excludes assymetrical share at the end
        # [:-1] becaus 'Category:'
        meta_dict.update({content_list[i][:-1] : content_list[i+1]})
        i = i+2
    """
    return meta_dict



def image_save_as_file_fn(url_image, file_name):
    """
    input: web url of jpg kind of image
    output: a file saved on local computer
    """
    response = requests.get(url_image)
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)
#Chose a random page
#url = "https://www.wikiart.org/en/francesco-clemente/the-four-corners-1985"

if __name__=="__main__":

    # Retrieve one painting
    url = "https://www.wikiart.org/en/raphael/vision-of-a-knight"

    print(get_meta_data(url))

    url_image=image_html_fn(url)

    file_name='sample.jpg'

    image_save_as_file_fn(url_image, file_name)
