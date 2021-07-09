import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import urllib.request as req
from urllib.parse import quote
from find.models import Result


'''
    언론사별 크롤링 실행 함수
'''

def munhwa(keyword):
    '''
        문화일보
        coded by : 김광휘
    '''
    xkwd = quote(keyword)
    url = 'http://mhsearch.munhwa.com/search.php?query='+xkwd
    html = urlopen(url)
    # print(url)
    soup = BeautifulSoup(html, 'html.parser')

    # title 추출
    title = soup.select_one('ul > li > div > a > strong').text
    # 날짜 추출
    date = soup.select_one('div > ul > li > div > a > span.data').text.strip()[-10:].replace("-",".")
    # link 추출
    link = soup.select_one('div > ul > li > div > a').attrs['href']
    # 요약 추출
    summary = soup.select_one('div > div > ul > li > div > a > span').text[:200]
    return '문화일보',title, date, link, summary
    # print(title)
    # print(date)
    # print(link)
    # print(summary)

def hankyoreh(keyword):
    '''
        한겨례 신문
        coded by 김광휘
    '''
    xkwd = quote(keyword)
    url = 'http://search.hani.co.kr/Search?command=query&keyword='+xkwd
    # url2 = '&media=news&submedia=&sort=s&period=all'
    html = urlopen(url)
    # print(url)
    soup = BeautifulSoup(html, 'html.parser')

    # title 추출
    lk = soup.select_one('li > dl > dt > a')
    title = lk.text
    # link 추출
    link = "http:" + str(lk.attrs['href'])
    # 날짜 추출
    date = soup.select_one('dl > dd.date > dl > dd').text[:10]
    # 요약 추출
    summary = soup.select_one('ul.search-result-list > li > dl > dd.detail').text.strip()[:200]
    return '한겨례신문',title, date, link, summary
    # print(title)
    # print(link)
    # print(dt)
    # print(smr)

def khann(keyword):
    '''
        경향신문
        coded by : 박세영
    '''
    xkwd = quote(keyword)
    url = 'http://search.khan.co.kr/search.html?stb=khan&q=' + xkwd

    req = requests.get(url + '&pg=' + '0' + '&sort=1')
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    n_title = soup.select_one('dl.phArtc > dt > a').text
    n_date = soup.select_one('dl.phArtc > dt > span.date').text[1:13]
    a_text = soup.select_one('dl.phArtc > dd.txt').text
    n_url = soup.select_one('dl.phArtc > dt > a').get('href')
    # print(n_title)
    # print(n_date)
    # print(a_text)
    # print(n_url)
    return '경향신문', n_title, n_date, n_url, a_text

def ohmyy(keyword):
    '''
        오마이뉴스
        coded by : 박세영
    '''
    xkwd = quote(keyword)
    url= 'http://www.ohmynews.com/NWS_Web/Search/s_news.aspx?keyword=' + xkwd
    url2 = '&order=DATE'
    # url2 = '&srh_area=1&srhdate=1&s_year=2021&s_month=2&s_day=3&e_year=2021&e_month=5&e_day=4&section_code=0&area_code=0&form_code=0'

    req = requests.get(url+url2)  # + url2 + '&page=' + '1')
    # req = requests.get(url1 + keyword + url2 + '&page=' + '1')
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    n_title = soup.select_one('div.cont > dl > dt > a').text
    n_date = soup.select_one('div.cont > p.source').text[-10:]
    a_text = soup.select_one('div.cont > dl > dd').text
    n_url = 'http://www.ohmynews.com' + soup.select_one('div.cont > dl > dt > a').get('href')
    # print(n_title)
    # print(n_date)
    # print(a_text)
    # print(n_url)
    return "오마이뉴스", n_title, n_date, n_url, a_text

def DongA(keyword):
    '''
        동아일보
        coded by : 박나현
    '''
    xkwd = quote(keyword)
    TARGET_URL_BEFORE_PAGE_NUM = "http://news.donga.com/search?p="
    TARGET_URL_BEFORE_KEWORD = '&query='
    TARGET_URL_REST = '&check_news=1&more=1&sorting=1&search_date=1&v1=&v2=&range=2'

    url = TARGET_URL_BEFORE_PAGE_NUM + TARGET_URL_BEFORE_KEWORD + xkwd + TARGET_URL_REST
    res = req.urlopen(url).read()
    soup = BeautifulSoup(res, 'html.parser')

    title = soup.select_one('div.searchContWrap > div.searchCont > div > div.t > p.tit > a').text
    contents = soup.select_one('div.searchContWrap > div.searchCont > div > div.t > p.txt > a').text
    link = soup.select_one('div.searchContWrap > div.searchCont > div > div.p > a').get('href')
    date = soup.select_one('div.searchContWrap > div.searchCont > div > div.t > p.tit > span').text[:10]
    date = date.replace("-",".")
    return "동아일보", title, date, link, contents

def jungang(keyword):
    '''
        중앙일보
        coded by : 서경보
    '''
    try:
        xkwd = quote(keyword)
        # keyword = input('키워드를 입력하세요: ')  # 키워드입력
        url = 'https://news.joins.com/Search/UnifiedSearch?Keyword=' + xkwd  # 중앙일보검색사이트에 내가 적은 키워드 더하기
        url2='&SortType=New&SearchCategoryType=UnifiedSearch&PeriodType=All&ScopeType=All&ImageType=All&JplusType=All&BlogType=All&ImageSearchType=Image&TotalCount=0&StartCount=0&IsChosung=False&IssueCategoryType=All&IsDuplicate=True&Page=1&PageSize=3&IsNeedTotalCount=True'
        html = urlopen(url+url2)
        bsobject = BeautifulSoup(html, "html.parser")  # 크롤링

        title = bsobject.select('#searchNewsArea > div.bd > ul > li > div > h2 > a')  # 기사제목추출 list로추출
        article_title = title[0].text  # 추출된 기사에서 제목추출
        article_url = title[0].get('href')  # 추출된 기사에서 url추출

        summary = bsobject.select('#searchNewsArea > div.bd > ul > li > div > span.lead')  # 기사요약추출 list추출
        article_summary = summary[0].text  # 기사요약추출에서 요약추출

        date = bsobject.select(
            '#searchNewsArea > div.bd > ul > li > div > span.byline > em')  # 날짜 list추출
        article_date = date[1].text[:10]  # 날짜


        # print('\n기사제목 : {0} \n\n기사내용 : {1}  \n\n기사날짜 : {2} \n\n기사주소 : {3}'.format(article_title, article_summary,
        #                                                                           article_date, article_url))
        return "중앙일보", article_title, article_date, article_url, article_summary

        # 기사가안나올때 에러생겨. 예외처리하는거 ex/ 기사없을떄 => none 출력
    except:
        print('없는 검색 결과 입니다.')



'''
    언론사별 크롤링 결과를 데이터베이스에 저장
'''
def save_result(df, kwd_id):

    for i in range(len(df)):
        Result(news=df.iloc[i][0], title=df.iloc[i][1], date=df.iloc[i][2], link=df.iloc[i][3], summary=df.iloc[i][4], keyword_id= kwd_id).save()













############# 테스트 코드 ##############
if __name__ == '__main__':
    print()