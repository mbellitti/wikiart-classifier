
import requests
from bs4 import BeautifulSoup
#finds the  html address

def image_html_fn(url):
    """
    input: url of Wikiart page, which has a single image
    output: web url of the source image
    """
    page_trial_wikiart = requests.get(url)

    page_trial_wikiart_html = BeautifulSoup(page_trial_wikiart.content, 'html.parser')
    search_image_line=page_trial_wikiart_html.find_all(class_='ms-zoom-cursor') 
    str_list=str(search_image_line[0])

    #used this class because all (?) the webpages seem to have a property that when you take your mouse it zooms in. 
    #This is how I am trying to find source image
    #print(search_image_line)

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
url = "https://www.wikiart.org/en/francesco-clemente/the-four-corners-1985"
url_image=image_html_fn(url)
file_name='sample.jpg'
image_save_as_file_fn(url_image, file_name)
