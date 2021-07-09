# import os,django
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")# project_name project name
# django.setup()
#
# from find.models import Keyword, Result
# import pandas as pd
#
#
# keyword = Keyword.objects.last()
# kwd_id = Keyword.objects.count()
#
#
# x= Result.objects.filter(keyword_id = kwd_id)
#
# datas = []
# columns = ['언론사', '제목', '날짜', 'URL', '요약']
# for i, text in enumerate(x):
#     datas.append([])
#     datas[i].append(text.news)
#     datas[i].append(text.title)
#     datas[i].append(text.date)
#     datas[i].append(text.link)
#     datas[i].append(text.summary)
# # print(datas)
# df = pd.DataFrame(datas,columns=columns)
# # print(df)
#
#
#
#
# contents = ''
# for i in range(len(df)):
#     # 신문사 + 기사제목 + 날짜
#     contents += ('<br>'+'<a style="background: #F9F7F6; border-left: 0.5em solid #26A69A; padding: 0.5em; font: HANYGO250;">'
#                  +'<a style="font-size: medium;font-weight:bolder;">'+'['+df.iloc[i][0]+']'+ '</a>'+'&nbsp;&nbsp;&nbsp;'
#                  +'<a style="font-size: large; font-weight: bolder">'+ df.iloc[i][1] + '</a>'+ str('&nbsp;&nbsp;'+ df.iloc[i][2] )+ '</a>')
#     contents += ('<div>'+ '&nbsp;&nbsp;&nbsp;' + '</div>')
#     # 내용
#     contents += ('<div>'+ df.iloc[i][4] + '</div>' + '<br>')
#     # link
#     contents += ('기사원문 : '+'<a href='+df.iloc[i][3]+'>'+df.iloc[i][3]+'</a>'+'<br>')
#
#
#
# #######################################################
#
# import smtplib  # 메일을 보내기 위한 라이브러리 모듈
# from email.mime.multipart import MIMEMultipart
# from email.mime.base import MIMEBase
# from email.mime.text import MIMEText
# from email import encoders
# #import Parser
# import os
#
# #df =pd.dataframe(Parser.keyword)
# #print(df)
# contents = ''
# for i in range(len(df)):
#     # 기사제목
#     contents += ('<h2 style="background: #F9F7F6; border-left: 0.5em solid #26A69A; padding: 0.5em; font: HANYGO250; font-weight: bolder;">' + df.iloc[i][1] + '</h2>' + '\n')
#     # 신문사
#     contents += ('<b style="font-size: large;">'+ df.iloc[i][0] + '</b>')
#     # 날짜
#     contents += str( '&nbsp;&nbsp;&nbsp;&nbsp;' + '<span>'+ df.iloc[i][2] + '</span>' )
#     # p.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT
#     # 내용
#     contents += ('<div>'+ df.iloc[i][4] + '</div>' + '<br>')
#     # link
#     contents += ('<a href='+df.iloc[i][3]+'>'+df.iloc[i][3]+'</a>')
#
# #print(contents)
#
# # -----------------------------------------------------------------
# # print(Parser.keyword)
# #fname = Parser.mk_word(Parser.keyword)
# #os.chdir('C:\workspace\\teamProject\\files')
# #fpath = os.path.join(os.getcwd(), fname)
#
# # 보내는 사람 정보
# me = "bbangnah@gmail.com"
# my_password = "NaHyeon328"
#
# # 로그인하기
# s = smtplib.SMTP_SSL('smtp.gmail.com')
# s.login(me, my_password)
#
# # 받는 사람 정보 리스트로 작성
# emails = ['mijuna2@naver.com', 'bbangnah@gmail.com']
#
# # 여러 사람에게 보낼 for 반복문 작성
# for you in emails:
#     # 메일 기본 정보 설정
#     msg = MIMEMultipart('alternative')
#     #msg['Subject'] = "키워드 검색 결과 입니다."
#     msg['Subject'] = "["+keyword+"]"+" "+"검색 결과입니다."  # 제목에 키워드 들어가기
#     msg['From'] = me
#     msg['To'] = you
#
#     # 메일 내용 쓰기
#     # content
#     part2 = MIMEText(contents, 'html')
#     msg.attach(part2)
#
#     # 파일 첨부하기
#     part = MIMEBase('application', "octet-stream")
#     with open(fname+'.docx', 'rb') as file:             # fpath -> fname
#         part.set_payload(file.read())
#     encoders.encode_base64(part)
#     part.add_header('Content-Disposition', "attachment", filename=fname+'.docx')  # fpath -> fname
#     msg.attach(part)
#
#     # 메일 발송 요청
#     s.sendmail(me, you, msg.as_string())
#
# # 전송완료 보고 / 서버 종료
# print('메일 전송이 완료되었습니다.')
# s.quit()
