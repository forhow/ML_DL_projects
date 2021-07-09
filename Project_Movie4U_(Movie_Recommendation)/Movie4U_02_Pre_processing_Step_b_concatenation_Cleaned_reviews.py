'''
    Pre-processing - Step. b

        ; Data 결측 및 중복 제거

    1. Setting
    
    2. Missing and Duplicated Value Processing
    
    3. File Merge
    
    3. File Save

'''

import pandas as pd

'''
    TEST Case
'''
# df_dup = pd.read_csv('./crawling/cleaned_review_2018.csv', index_col=0)
# print(df_dup.info())
# df_dup.dropna(inplace=True)
# print(df_dup.info())
# df_undup = df_dup.drop_duplicates()
# print(df_undup.info())
# print(df_undup.duplicated().sum())
# df_undup.to_csv('./crawling/cleaned_review_2018.csv')


'1. Setting'
# 1. 병합용 Dataframe 생성
df_total = pd.DataFrame(columns = ['titles', 'cleaned_reviews'])


'2. Missing and Duplicated Value Processing'
for i in range(18,22):
    # 병합 대상 csv file load
    df_temp = pd.read_csv('./crawling/cleaned_review_20{}.csv'.format(i), index_col=0)

    # 결측 제거
    df_temp.dropna(inplace=True)

    # 중복제거
    df_temp.drop_duplicates(inplace=True)

    # 컬럼지정 및 파일 임시저장
    df_temp.columns = ['titles', 'cleaned_reviews']
    df_temp.to_csv('./crawling/cleaned_review20{}_concat.csv'.format(i))

    '3. File Merge'
    df_total = pd.concat([df_total, df_temp], ignore_index=True)


'4. File Save'
# 반복문 종료 후 최종 저장 Dataframe을 csv로 저장
df_total.to_csv('./crawling/cleaned_review_2018~2021.csv')
print(df_total.info())

