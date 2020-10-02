# %%
import requests
import pandas as pd
url = "https://quality.data.gov.tw/dq_download_csv.php?nid=9634&md5_url=7b41ec11a497a37184b82402be86eda5"
#res = requests.get(url)
# %%
df = pd.read_csv(url)
# %%
