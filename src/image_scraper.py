import requests
from bs4 import BeautifulSoup
import json


def get_meta_data_json(url):
    """Downloads the metadata of an artwork and returns a json object.

    Input:
        url: the url of an 'artwork' page

    Ouput:
        painting_json: a json file containing all the metadata available
        in the paintingJson attribute.

    Description:
        The source of 'artwork' pages contain an attribute 'paintingJson', which
        has many useful keys, including a direct url of the image.

        For example:

        {'_t': 'PaintingForGalleryNew',
        '_id': '5772847bedc2cb3880fded05',
        'title': 'Self-portrait',
        'year': '1574',
        'width': 423,
        'height': 600,
        'artistName': 'Hans von Aachen',
        'image': 'https://uploads4.wikiart.org/images/hans-von-aachen/self-portrait-1574.jpg',
        'map': '0123**67*',
        'paintingUrl': '/en/hans-von-aachen/self-portrait-1574',
        'artistUrl': '/en/hans-von-aachen',
        'albums': None,
        'flags': 4,
        'images': None}
    """

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'lxml')

    # this string is of the form "paintingJson = {"att": "val"}"
    # so we need to split it before parsing it
    painting_json_string = soup.find(class_="wiki-layout-painting-info-bottom")['ng-init']

    meta_data = json.loads(painting_json_string.split("=")[1])

    ##################################################
    # Merging wit Mohit version
    ##################################################

    article = soup.find('article')
    catgs = [cat.string.strip(':') for cat in article.find_all('s')]

    ##################################################
    # Done up to here
    ##################################################

    # Get rid of stuff that causes us troubles (i.e. pure text we don't need):
    #   *   <script> tag
    #   *   'order reproduction' box
    # If you find more exception that need to be handled this is a good place
    # to integrate them into the code.
    if article.script is not None:
        article.script.extract()
    if article.find_next(class_ = 'order-reproduction') is not None:
        article.find_next(class_ = 'order-reproduction').extract()

    # Retrieve remaining strings
    content_list = list(article.stripped_strings)

    # Create dictionary for metadata from this list using categories.
    # The first two lines of text left are title and author.
    #   --> always true??
    meta_dict = {'url': url,
        'title' : content_list[0], 'artist' : content_list[1]}

    # Get content list as a single string
    #
    # last category is 'Share'; we don't want to share
    content_str = ''.join(content_list)
    for catg_idx in range(len(catgs)-1):
        beg = content_str.find(catgs[catg_idx]) + len(catgs[catg_idx]) +1
        end = content_str.find(catgs[catg_idx + 1])
        meta_dict.update({catgs[catg_idx]: content_str[beg:end]})

    return meta_data


def image_html_fn(url):
    """Extracts the image url from an 'artwork' page.

    Input:
        url: the URL of an 'artwork' Wikiart page.

    Output:
        img_url: direct url of the source image.

    See Also:
        This same information is contained in the 'image' key of the JSON object
        returned by `get_meta_data_json`.
    """

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'lxml')

    # the image url is in the "open graph" section
    img_url = soup.find(attrs={"property":"og:image"})['content']

    assert img_url,"Image not found."

    return img_url


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
    soup = BeautifulSoup(page.content, 'lxml')
    article = soup.find_all('article')
    assert len(article) == 1, "Found more than one 'article' tag on page"
    article = article[0]

    # get metadata categories
    catgs = article.find_all('s')

    # Get actual text strings and getting rid of colon at the end of category.
    for i in range(len(catgs)):
        catgs[i] = str(catgs[i].string)[:-1]

    # Get rid of stuff that causes us troubles (i.e. pure text we don't need):
    #   *   <script> tag
    #   *   'order reproduction' box
    # If you find more exception that need to be handled this is a good place
    # to integrate them into the code.
    if article.script is not None:
        article.script.extract()
    if article.find_next(class_ = 'order-reproduction') is not None:
        article.find_next(class_ = 'order-reproduction').extract()

    # Retrieve remaining strings
    content_list = list(article.stripped_strings)

    # Create dictionary for metadata from this list using categories.
    # The first two lines of text left are title and author.
    #   --> always true??
    meta_dict = {'url': url,
        'title' : content_list[0], 'artist' : content_list[1]}

    # Get content list as a single string
    #
    # last category is 'Share'; we don't want to share
    content_str = ''.join(content_list)
    for catg_idx in range(len(catgs)-1):
        beg = content_str.find(catgs[catg_idx]) + len(catgs[catg_idx]) +1
        end = content_str.find(catgs[catg_idx + 1])
        meta_dict.update({catgs[catg_idx]: content_str[beg:end]})

    return meta_dict


# TODO: what happens if the image already exists?
def image_save_as_file_fn(img_url, file_name=None):
    """Downloads an image and saves it to disk.

    Input:
        img_url: Direct url of an image

        file_name: optional file name to save the image to. If not
        given the one contained in the URL is used.
    """
    response = requests.get(url_image,stream=True)

    # the URL is something like http://stuff.com/image.jpg
    if file_name is None:
        file_name = img_url.split('/')[-1]

    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)


if __name__=="__main__":

    # Retrieve one painting as example
    page_url = "https://www.wikiart.org/en/raphael/vision-of-a-knight"

    print(get_meta_data(page_url))

    url_image=image_html_fn(page_url)

    print(url_image)

    # file_name='sample.jpg'

    image_save_as_file_fn(url_image)
