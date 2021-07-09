'''
    Crawling - Naver movie Reviews
    
     - by Selenium with Chrome Driver
     
    1. Setting 
    
    2. Entry of Objective year and the number of movies
    
    3. Crawling
     a. Crawling data save - Fail Backup
     b. Crawling data save - Main method
    
'''

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import requests
import time
import csv


# 'Crawling TEST'
# url ='https://movie.naver.com/movie/sdb/browsing/bmovie.nhn?open=2019&page=1'
# driver.get(url)
# driver.find_element_by_xpath('//*[@id="old_content"]/ul/li[1]/a').click()
# time.sleep(0.5)
# driver.find_element_by_xpath('//*[@id="movieEndTabMenu"]/li[6]/a/em').click()
# driver.find_element_by_xpath('//*[@id="reviewTab"]/div/div/ul/li[1]').click()


'1. Setting'

# Browser Options
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('disable_gpu')
options.add_argument('lang=ko_KR')

# Driver
driver = webdriver.Chrome('chromedriver', options=options)
titles = []
reviews = []


'2. Entry of Objective year and the number of movies'
# Crawling 대상 정보 입력
year = int(input('크롤링 대상 년도 입력 (ex. 2021): '))
page = int(input('해당 연도의 영화개수 입력 (ex. 608): '))
page = (page // 20) +2


'3. Crawling'
# Crawling 수행
try:
    # 년도별 개봉영화 목록 페이지
    for i in range(1, page):
        url = 'https://movie.naver.com/movie/sdb/browsing/bmovie.nhn?open={}&page={}'.format(year, i)

        # 페이지당 영화 목록 수 (20개)
        for j in range(1,21):
            try:
                # 해당 년도 영화목록 페이지 접속
                driver.get(url)
                time.sleep(1)

                # 영화 클릭 (영화 상세페이지 이동)
                movie_title_xpath = '//*[@id="old_content"]/ul/li[{}]/a'.format(j)
                title = driver.find_element_by_xpath(movie_title_xpath).text
                print(title)
                driver.find_element_by_xpath(movie_title_xpath).click()
                time.sleep(1)
                try:
                    # 영화 상세 페이지에서 리뷰버튼 클릭
                    btn_review_xpath = '//*[@id="movieEndTabMenu"]/li[6]/a/em'
                    driver.find_element_by_xpath(btn_review_xpath).click()
                    time.sleep(1)

                    # 리뷰 개수 확인 및 리뷰 페이지 계산
                    review_len_xpath = '//*[@id="reviewTab"]/div/div/div[2]/span/em'
                    review_len = driver.find_element_by_xpath(review_len_xpath).text
                    review_len = int(review_len.replace(',',''))

                    # 리뷰개수를 50개로 제한
                    if review_len > 50:
                        review_len = 50

                    try:
                        # 리뷰 페이지 선택
                        for k in range(1, ((review_len-1) // 10)+2):
                            review_page_xpath = '//*[@id="pagerTagAnchor{}"]/span'.format(k)
                            driver.find_element_by_xpath(review_page_xpath).click()
                            time.sleep(1)

                            # 리뷰 선택 및 크롤링 수행
                            for l in range(1,11):
                                review_title_xpath = '//*[@id="reviewTab"]/div/div/ul/li[{}]'.format(l)
                                try:
                                    # 리뷰 선택
                                    driver.find_element_by_xpath(review_title_xpath).click()
                                    time.sleep(1)
                                    try:
                                        # 영화제목 및 리뷰 크롤링
                                        review_xpath = '//*[@id="content"]/div[1]/div[4]/div[1]/div[4]'
                                        review = driver.find_element_by_xpath(review_xpath).text

                                        '''
                                            3.a. Crawling data save - Fail Backup
                                            
                                            크롤링 수행 시 결과를 2가지 방법으로 저장.
                                            
                                            backup 선택하지 않을 경우 Line 126~131 주석처리 후 실행 바랍니다.
                                        '''

                                        # main : 리스트에 내용 추가 시킨 후 한번에 Dataframe과 파일로 저장
                                        titles.append(title)
                                        reviews.append(review)

                                        # backup : 크롤링 성공시마다 파일에 직접 추가
                                        append_data = {'years': year, 'titles': title, 'reviews': review}
                                        with open('./crawling/reviews_{}_backup.csv'.format(year), 'a', encoding='utf-8', newline='') as save:
                                            fieldnames = ['years', 'titles', 'reviews']
                                            writer = csv.DictWriter(save, fieldnames=fieldnames)
                                            writer.writerow(append_data)

                                        driver.back()
                                        time.sleep(1)
                                    except:
                                        driver.back()
                                        time.sleep(1)
                                        print('review crawling error')
                                except:
                                    time.sleep(1)
                                    print('review title click error')
                    except:
                        print('review page btn click error')
                except:
                    print('review btn click error')

            except NoSuchElementException:
                driver.get(url)
                time.sleep(1)
                print('NoSuchElementException')
        print(len(reviews))

    '3.b. Crawling data save - Main method'
    # main : 리스트에 내용 추가 시킨 후 한번에 Dataframe과 파일로 저장
    # 크롤링 결과 Dataframe 생성
    df_review = pd.DataFrame({'titles':titles, 'reviews':reviews})
    df_review['years'] = year
    print(df_review.head(20))
    # CSV 파일 저장
    df_review.to_csv('./crawling/reviews_{}.csv'.format(year))


except:
    print('except1')
finally:
    driver.close()


'''
    수정 조치사항
    - 영화별 / 페이지별로 저장되도록 수정 권고 -> 현재 리뷰별로 백업 저장 구축완료
    - years column 제거 -> 기 진행된 파일과의 일관성 위해 유지, 전처리 시 조치
'''

















