import json
import requests
from bs4 import BeautifulSoup


def get_meta_data(url,wanted_keys=None):
    """Downloads the metadata of an artwork and returns a json object.

    Input:
        url: the url of an 'artwork' page

        wanted_keys: None,"all" or a list of strings.
        List of keys in the dictionary associated to the artwork.
        If None, returns the default metadata list
            ['_id','title','artistname','image',
            'date','style','genre']
        If "all" returns a whole lot of other metadata scraped from
        the page. Other available keys include:
            'original title','height','width','media','location',
            'dimensions'
        A list of strings makes it return only the one you asked for.
        If the page doesn't have one of the keys requested, it will
        have value None.

    Ouput:
        meta_data: a dictionary file containing the requested painting
        metadata.

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

    # This string is of the form "paintingJson = {"att": "val"}"
    painting_json_string = soup.find(class_="wiki-layout-painting-info-bottom")['ng-init']
    meta_data = json.loads(painting_json_string.lower().split("=")[1])

    # All the information we need is in the 'li' tag, but there are also some useless things
    for lis in soup.find('article').find_all('li'):
        key,*val = lis.stripped_strings
        meta_data[key.strip(':').lower()] = "".join(val).lower()


    if wanted_keys == "all":
        return meta_data
    else:
        if wanted_keys is None:
            wanted_keys = ['_id','title','artistname','image','date','style','genre']
        return {key:meta_data[key] if key in meta_data else None for key in wanted_keys}


def image_html_fn(url):
    """Extracts the image url from an 'artwork' page.

    This same information is contained in the tag 'image' of the
    dictionary returned by get_meta_data().

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

    print("Default metadata:")

    print(get_meta_data(page_url))

    print("All metadata:")

    print(get_meta_data(page_url,wanted_keys="all"))

    url_image=image_html_fn(page_url)

    print(url_image)

    # file_name='sample.jpg'

    # image_save_as_file_fn(url_image)