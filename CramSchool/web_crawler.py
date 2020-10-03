# %%
import pandas as pd
import requests
from bs4 import BeautifulSoup
from time import sleep
url = "https://data.gov.tw/dataset/16461"  # 各縣市短期補習班


def get_citys_link(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text)
    data_download_div = soup.find("div", {
                                  "class": "field field-name-field-dataset-resource field-type-dgresource-resouce field-label-inline clearfix"})
    # for loop + append all even citys
    data_links_even = data_download_div
    Citys = []
    Links = []
    for i in range(15):
        try:
            data_links_even = data_links_even.findNext(
                "div", {"class": "field-item even"})
            city_span = data_links_even.find("span", {"class": "ff-desc"})
            link_a = data_links_even.find(
                "a", {"class": "dgresource ff-icon ff-icon-json"})
            Citys.append(city_span.text[:3])
            Links.append(link_a.attrs["href"])
            sleep(1)
        except:
            print("No more city!")
    # for loop + append all odd citys
    data_links_odd = data_download_div
    for i in range(15):
        try:
            data_links_odd = data_links_odd.findNext(
                "div", {"class": "field-item odd"})
            city_span = data_links_odd.find("span", {"class": "ff-desc"})
            link_a = data_links_odd.find(
                "a", {"class": "dgresource ff-icon ff-icon-json"})
            Citys.append(city_span.text[:3])
            Links.append(link_a.attrs["href"])
            sleep(1)
        except:
            print("No more city!")
    # make a dict
    if len(Citys) == len(Links):
        Citys_Link = {Citys[i]: Links[i] for i in range(len(Citys))}
        print(f"We got {len(Citys)} citys.")
    else:
        print("length of Citys is not equal to length of Links!")
    return Citys_Link
#  concat all dataframe of city


def json_to_DataFrame(url):
    Citys_Link = get_citys_link(url)
    df = pd.DataFrame({})
    for k, v in Citys_Link.items():
        print(k)
        print(v)
        sleep(15)  # 必須設置時間間隔才不會被拉黑
        df_tmp = pd.read_json(v)
        df = pd.concat([df, df_tmp])
    return df


# %%
json_to_DataFrame(url)
# %%
