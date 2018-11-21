"""A module to retrieve artworks from WikiArt and manipulate them."""

import json
import os.path
import requests
from bs4 import BeautifulSoup

class ArtWork:
    """Class to represent artworks.

    During initialization it retrieves the metadata from
    WikiArt and stores them in the attributes.

    If you want to see all the available attributes, you can either
    check `art.__dict__` or just do `print(art)`.

    Examples:
        art = ArtWork("http://some_painting_on_wikiart")
        art.title
    """

    def __init__(self,url):
        self.__dict__ = get_meta_data(url)

    def __str__(self):
        s = "\n"
        for key,val in self.__dict__.items():
            s += ("{}: {}\n".format(key,val))

        return s

    def get_image(self):
        """Download the image of the artwork, unless it
        already exists.

        The name of the image file is saved
        in the `imagefile` attributeself.

        Exmples:
            art = ArtWork(url)
            art.get_image()
        """

        self.imagefile = download_image(self.image)

def get_meta_data(url,wanted_keys=None):
    """Downloads the metadata of an artwork and returns a json object.

    Arguments:
        url: the url of an 'artwork' page
        wanted_keys: None,"all" or a list of strings.
            List of keys in the dictionary associated to the artwork.
            If None, returns the default metadata list
                ['_id','title','artistname','image',
                'year','style','genre']
            If "all" returns a whole lot of other metadata scraped from
            the page. Other available keys include:
                'original title','height','width','media','location',
                'dimensions'
            A list of strings makes it return only the one you asked for.
            If the page doesn't have one of the keys requested, it will
            have value None.

    Returns:
        meta_data: a dictionary file containing the requested painting
            metadata.

    Examples:
        page_url = "https://www.wikiart.org/en/raphael/vision-of-a-knight"
        data = get_meta_data(page_url)
        {'_id': '577271fdedc2cb3880c3846a',
         'title': 'vision of a knight',
         'artistname': 'raphael',
         'image': 'https://uploads5.wikiart.org/images/raphael/vision-of-a-knight.jpg',
         'year': '1504',
         'style': 'high renaissance',
         'genre': 'genre painting'}
    """

    print("Retrieving "+url)

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'lxml')

    # This string is of the form "paintingJson = {"att": "val"}"
    painting_json_string = soup.find(class_="wiki-layout-painting-info-bottom")['ng-init']
    meta_data = json.loads(painting_json_string.lower().split("=",1)[1])

    # All the information we need is in the 'li' tag, but there are also some useless things
    for lis in soup.find('article').find_all('li'):
        key,*val = lis.stripped_strings
        meta_data[key.strip(':').lower()] = "".join(val).lower()


    if wanted_keys == "all":
        return meta_data
    else:
        if wanted_keys is None:
            wanted_keys = ['_id','title','artistname','image','year','style','genre']
        return {key:meta_data[key] if key in meta_data else None for key in wanted_keys}


def download_image(img_url, file_name=None):
    """Downloads an image and saves it to disk.

    Arguments:
        img_url: Direct url of an image
        file_name: optional file name to save the image to. If not
            given the one contained in the URL is used.

    Returns:
        file_name: file name of the downloaded image
    """

    # the URL is something like http://stuff.com/image.jpg
    if file_name is None:
        file_name = img_url.split('/')[-1]

    if os.path.isfile(file_name):
        print("File already exists: {}".format(file_name))
    else:
        response = requests.get(img_url,stream=True)

        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
        # TODO: Raise an exception if the request is bad?

    return file_name


# DO NOT USE THIS. Seriously. Do not.
def image_html_fn(url):
    """Extracts the image url from an 'artwork' page.

    This same information is contained in the tag 'image' of the
    dictionary returned by get_meta_data(), please use the version of this
    function that doesn't make an extra request.

    Arguments:
        url: the URL of an 'artwork' Wikiart page.

    Returns:
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


if __name__=="__main__":

    # Retrieve one painting as example
    # page_url = "https://www.wikiart.org/en/raphael/vision-of-a-knight"
    # page_url = "https://www.wikiart.org/en/hans-von-aachen/self-portrait-1574"
    page_url = "https://www.wikiart.org/en/giovanni-bellini/the-feast-of-the-gods-1514"

    print("Default metadata:")

    print(get_meta_data(page_url))

    print("All metadata:")

    print(get_meta_data(page_url,wanted_keys="all"))

    url_image=image_html_fn(page_url)

    print(url_image)

    # file_name='sample.jpg'

    # image_save_as_file_fn(url_image)
