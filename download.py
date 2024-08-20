import os
import pandas as pd
import wget
import random
import urllib

txt_file_path='/home/yichao/Desktop/multimedia_marginatum.txt'
dir='/home/yichao/Desktop/New Folder'
data=pd.read_csv(txt_file_path,delim_whitespace = True, engine='python')
link=data['identifier'].values
for i in link:
    try:
        file_name=wget.download(i, out=os.path.join(dir,'{name}.jpg'.format(name=random.randint(10000000,99999999))))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"File not found for URL: {i}")
