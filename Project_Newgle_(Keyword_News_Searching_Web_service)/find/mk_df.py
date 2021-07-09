from find import Search
import pandas as pd

def dataframe(keyword):
    mh = Search.munhwa(keyword)
    mh_news = mh[0]
    mh_title = mh[1]
    mh_date = mh[2]
    mh_link = mh[3]
    mh_summary = mh[4]

    hkr = Search.hankyoreh(keyword)
    hkr_news = hkr[0]
    hkr_title = hkr[1]
    hkr_date = hkr[2]
    hkr_link = hkr[3]
    hkr_summary = hkr[4]

    khn = Search.khann(keyword)
    khn_news = khn[0]
    khn_title = khn[1]
    khn_date = khn[2]
    khn_link = khn[3]
    khn_summary = khn[4]

    omy = Search.ohmyy(keyword)
    omy_news = omy[0]
    omy_title = omy[1]
    omy_date = omy[2]
    omy_link = omy[3]
    omy_summary = omy[4]

    da = Search.DongA(keyword)
    da_news = da[0]
    da_title = da[1]
    da_date = da[2]
    da_link = da[3]
    da_summary = da[4]

    ja = Search.jungang(keyword)
    ja_news = ja[0]
    ja_title = ja[1]
    ja_date = ja[2]
    ja_link = ja[3]
    ja_summary = ja[4]

    df_list = [mh, da, ja, hkr, khn, omy]
    columns = ['언론사', '제목', '날짜', 'URL', '요약']

    df = pd.DataFrame(df_list, columns=columns)