import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option("display.max_column",None)




df_ = pd.read_excel("data/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]




today_date = dt.datetime(2011, 12, 11)
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# monetary
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# convert recency ve T values to weekly period
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

#select more than 1 frequency values

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

#select more than 0 monetary values
cltv_df = cltv_df[cltv_df["monetary"] > 0]


#BG-NBD Model
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

#GAMMA-GAMMA Model

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

# CLTV prediction for 6 months

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv = cltv.reset_index()

cltv_df = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_df.head()

