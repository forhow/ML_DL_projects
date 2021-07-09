'''
    Project : Movie4U  - 영화 리뷰 분석 및 키워드 기반 추천 시스템

    - Data 수집 대상기간 : 개봉년도 기준 2018 ~ 2021

    - Data 수집처 : Naver Movie

    - Data 수집 방법 : Crawling with Selenium

    - 수집 대상 Data
     a. 영화별 개봉년도
     b. 영화제목
     c. 영화에 대한 리뷰

    - Pre-processing
     a. 영화에 대한 리뷰 내용 전처리
        ; 형태소 분석, 불용어 제거 등 리뷰 문장에 대한 가공
     b. 영화 별 리뷰 항목 전처리
        ; 전처리 step a. 완료 파일들의 병합
     c. 데이터 중복 및 결측 제거
        ; 영화 별 모든 리뷰문장을 병합, 영화 당 하나의 리뷰문장 목록으로 구성되도록 가공

    - TF-IDF Vectorization : 문장간 유사도 측정, Matrix 작성

    - Vectorization, Modeling : Word2Vec 사용
        ; Embedding Modeling

    - Recommendation Program
        ; Keyword 입력받아 리뷰의 문장유사도가 가장 큰 영화 10개 선정 추천

    - Reference
        ; Visualization : Word Cloud
        ; Visualivation : keyword와 연관단어간의 배치


'''

※ Task Request Sample
'''
    수행작업명 : Crawling

    수행방법 :
        1. 크롤링 대상년도 입력 및 크롤링 진행
        2. 크롤링 완료 데이터(.csv) upload
        3. 데이터 종합해서 Raw data 생성

    작업/완료 데이터 형식 : Pandas - Dataframe
     - Columns : ['years','titles', 'reviews']

    작업/완료 파일 형식: .csv
    파일명 : reviews_0000.csv

    코드/결과물 저장소 : Google Drive
'''
