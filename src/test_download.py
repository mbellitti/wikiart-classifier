import pandas as pd
import os.path


urls_large = pd.read_csv('img_urls_large.txt', header = None)
a = []
for i in range(len(urls_large)):
    if os.path.isfile('images\\' + urls_large.iloc[i][0].split(' ')[0] +'.jpg'):
        a.append(i)
    if i%100 == 0:
        print(i)
print(len(a))

url_large_rest = urls_large.drop(a)
url_large_rest.to_csv("img_urls_rest.txt",sep=" ",header=False,index=False)
