
from datetime import datetime
import requests
import json
import pandas as pd
import pytz


def get_currentprice(urlPrice, symbol, resolution):
    currentPrice = requests.get(urlPrice)
    priceParsed = json.loads(currentPrice.text)
    data_time = pd.to_datetime(priceParsed["t"], unit="s", utc=True).tz_convert("Asia/Ho_Chi_Minh")
    data_time = data_time.tz_localize(None).floor('us')
    data_date = pd.to_datetime(data_time.date)
    df = pd.DataFrame(
        {
            "data_date": data_date,
            # data_time.tz_convert("Asia/Ho_Chi_Minh").date,
            "resolution": resolution,
            "ticker": symbol,
            "open_price": priceParsed["o"],
            "highest_price": priceParsed["h"],
            "lowest_price": priceParsed["l"],
            "close_price": priceParsed["c"],
            "volume": priceParsed["v"],
            "created_at": pd.to_datetime(datetime.now()),
            "updated_at": pd.to_datetime(datetime.now())
        }
    )
    return df


def aggregate_data(startdate, enddate, resolution, urlPrice, tickers):
    aggregated_data = []
    for ticker in tickers:
        startdate_int = int(startdate.timestamp())
        enddate_int = int(enddate.timestamp())
        url = urlPrice + "symbol=" + ticker + "&resolution=" + resolution + "&from=" + str(
            startdate_int) + "&to=" + str(enddate_int)
        price = get_currentprice(url, ticker, resolution)
        aggregated_data.append(price)
    final_df = pd.concat(aggregated_data, ignore_index=True)
    return final_df


def process(startdate, enddate, resolution, urlPrice):
    tickers = [
        "AAA", "ACB", "ANV", "BCG", "BCM", "BID", "BMP", "BSI", "BVH", "BWE",
        "CII", "CMG", "CTD", "CTG", "CTR", "CTS", "DBC", "DCM", "DGC", "DGW",
        "DIG", "DPM", "DSE", "DXG", "DXS", "EIB", "EVF", "FPT", "FRT", "FTS",
        "GAS", "GEX", "GMD", "GVR", "HAG", "HCM", "HDB", "HDC", "HDG", "HHV",
        "HPG", "HSG", "HT1", "IMP", "KBC", "KDC", "KDH", "KOS", "LPB", "MBB",
        "MSB", "MSN", "MWG", "NAB", "NKG", "NLG", "NT2", "OCB", "PAN", "PC1",
        "PDR", "PHR", "PLX", "PNJ", "POW", "PPC", "PTB", "PVD", "PVT", "REE",
        "SAB", "SBT", "SCS", "SHB", "SIP", "SJS", "SSB", "SSI", "STB", "SZC",
        "TCB", "TCH", "TLG", "TPB", "VCB", "VCG", "VCI", "VGC", "VHC", "VHM",
        "VIB", "VIC", "VIX", "VJC", "VND", "VNM", "VPB", "VPI", "VRE", "VTP"
    ]
    df = aggregate_data(startdate, enddate, resolution, urlPrice, tickers)
    df.to_csv("VN100_stock_price_1D.csv", index=False)

urlPrice = "https://banggia.tvs.vn/datafeed/history?"
timezone = pytz.timezone('Asia/Bangkok')

today = datetime.now(timezone)
# startdate = datetime(today.year, today.month, today.day, 8, 0, 0, tzinfo=timezone)
# enddate = datetime(today.year, today.month, today.day, 16, 0, 0, tzinfo=timezone)
startdate = datetime(2018,1,1,8,0,0,tzinfo=timezone)
enddate = datetime(2025,3,22 ,16,0,0,tzinfo=timezone)
# process(startdate, enddate, "1", urlPrice)
# process(startdate,enddate,"5",urlPrice)
# process(startdate,enddate,"15",urlPrice)
# process(startdate,enddate,"30",urlPrice)
# process(startdate,enddate,"60",urlPrice)
process(startdate,enddate,"1D",urlPrice)