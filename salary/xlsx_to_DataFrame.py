# %%
from datetime import datetime
import pandas as pd
# 讀取df
df = pd.read_excel("Industry_Salary.xlsx")
# 欄位名稱去掉中文
df.columns = list(map(lambda x: "_".join(x.split("_")[1:]), df.keys()))
# 紀錄最後一個年月
last_year_month = df["Year_and_month"][len(df)-1][:6]
# 丟棄整年度的平均值
first_year = int(df["Year_and_month"][0][:4])
last_year = int(df["Year_and_month"][len(df)-1][:4])
df = df.drop([13*i for i in range(last_year-first_year+1)], axis=0)  # 丟棄列
# 將年月別改成時間格式
df["Year_and_month"] = list(
    map(lambda x: datetime.strptime(x, "%Y%m"), df["Year_and_month"]))
# "-"改成0
df = df.replace("-", 0)
# 空值補0
df = df.fillna(0)
# 將所有值改成int
for col in df.keys():
    if col == "Year_and_month":
        print("OK")
    else:
        df[col] = list(map(lambda x: int(x), df[col]))
# 轉tidy形式
df = pd.melt(df, id_vars=df.columns[:1], value_vars=df.columns[1:],
             var_name="Industry", value_name="Salary")
# 輸出csv
df.to_csv("Salary_1980_"+last_year_month+".csv", encoding='utf-8', index=False)
# %%
