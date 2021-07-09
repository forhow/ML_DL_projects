"""
    project NewsCategory4U

    - News category classifier

    - beautifulsoup 사용
"""

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

'setting'
pd.set_option('display.unicode.east_asian_width', True)

# category 지정
category = 'Politics Economic Social Culture World IT'.split()

# url 규칙 확인
#경제 https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=101#&date=%2000:00:00&page=1
#사회 https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=102#&date=%2000:00:00&page=1
url = 'https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=100#&date=%2000:00:00&page='
headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36'}

# category별 page 수
page_num = [334, 423, 400, 87, 128, 74]

# 저장할 dataframe 생성
df_section_title = pd.DataFrame()


'Crawling'
for j in range(0, 6):
  url = 'https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=10{}#&date=%2000:00:00&page='.format(j)
  print(category[j])
  for i in range(1, page_num[j]):
    resp = requests.get(url+str(i), headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    title_tags = soup.select('.cluster_text_headline')
    titles = []
    for title_tag in title_tags:
      titles.append(re.compile('[^가-힣 | a-z | A-Z | 0-9]+').sub(' ', title_tag.text))
    df = pd.DataFrame(titles, columns = ['title'])
    df['category'] = category[j]
    # progress status display
    print('.', end='')
    if i % 50 ==0 :
      print()
    df_section_title = pd.concat([df_section_title, df], axis='rows', ignore_index=True)

print(df_section_title[:50])
print(df_section_title.info())


'Data save'
df_section_title.to_csv('../datas/news_titles_data.csv', encoding='utf-8-sig')

