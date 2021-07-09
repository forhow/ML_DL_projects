import smtplib  # 메일을 보내기 위한 라이브러리 모듈
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import os,django
from django.shortcuts import redirect, render , HttpResponseRedirect
from find.forms import EmailForm
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")# project_name project name
django.setup()
from find.models import Keyword, Result
import pandas as pd
from django.urls import reverse


def mail_sender(request):
    if request.method == "POST":
        #form = EmailForm(request.POST)
        e_mail = request.POST['mail_sender']

        import os, django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")  # project_name project name
        django.setup()
        # print(keyword)
        kwd_id = Keyword.objects.count()
        keyword = Keyword.objects.get(id=kwd_id)
        print(keyword)


        x= Result.objects.filter(keyword_id = kwd_id)

        datas = []
        columns = ['언론사', '제목', '날짜', 'URL', '요약']
        for i, text in enumerate(x):
            datas.append([])
            datas[i].append(text.news)
            datas[i].append(text.title)
            datas[i].append(text.date)
            datas[i].append(text.link)
            datas[i].append(text.summary)
        # print(datas)
        df = pd.DataFrame(datas,columns=columns)
        # print(df)


        contents = ''
        for i in range(len(df)):
            # 신문사 + 기사제목 + 날짜
            if i <= 2:
                # 보수 신문사 + 기사제목 + 날짜
                contents += (
                            '<br>' + '<p style="background: #F9EBEA; border-left: 0.5em solid #E74C3C; padding: 0.5em; font: HANYGO250;">'
                            + '<a style="font-size: medium;font-weight:bolder;">' + '[' + df.iloc[i][
                                0] + ']' + '</a>' + '&nbsp;&nbsp;&nbsp;'
                            + '<a style="font-size: large; font-weight: bolder">' + df.iloc[i][1] + '</a>' + str(
                        '&nbsp;&nbsp;' + df.iloc[i][2]) + '</p>')
            else:
                # 진보 신문사 + 기사제목 + 날짜
                contents += (
                            '<br>' + '<p style="background: #E8EAF6; border-left: 0.5em solid #3F51B5; padding: 0.5em; font: HANYGO250;">'
                            + '<a style="font-size: medium;font-weight:bolder;">' + '[' + df.iloc[i][
                                0] + ']' + '</a>' + '&nbsp;&nbsp;&nbsp;'
                            + '<a style="font-size: large; font-weight: bolder">' + df.iloc[i][1] + '</a>' + str(
                        '&nbsp;&nbsp;' + df.iloc[i][2]) + '</p>')

            contents += ('<div>' + '&nbsp;&nbsp;&nbsp;' + '</div>')
            # 내용
            contents += ('<div>' + df.iloc[i][4] + '</div>' + '<br>')
            # link
            contents += ('기사원문 : ' + '<a href=' + df.iloc[i][3] + '>' + df.iloc[i][3] + '</a>' + '<br>')

        # 보내는 사람 정보
        '''
                SMTP SERVER 사용자 주소 / 비밀번호 입력필요
        '''
        me = "SENDER_ADDRESS@YOUR.EAMIL"
        my_password = "SENDER PASSWORD"

        # 로그인하기
        s = smtplib.SMTP_SSL('smtp.gmail.com')
        s.login(me, my_password)

        # 받는 사람 정보 리스트로 작성
        # emails = ['mijuna2@naver.com', 'bbangnah@gmail.com']
        emails = [e_mail]


        # 여러 사람에게 보낼 for 반복문 작성
        for you in emails:
            # 메일 기본 정보 설정
            msg = MIMEMultipart('alternative')
            #msg['Subject'] = "키워드 검색 결과 입니다."
            msg['Subject'] = "[" + str(keyword) + "]" + " " + "검색 결과입니다."  # 제목에 키워드 들어가기
            msg['From'] = me
            msg['To'] = you

            # 메일 내용 쓰기
            # content
            part2 = MIMEText(contents, 'html')
            msg.attach(part2)

            # # 파일 첨부하기
            # part = MIMEBase('application', "octet-stream")
            # with open(fname+'.docx', 'rb') as file:             # fpath -> fname
            #     part.set_payload(file.read())
            # encoders.encode_base64(part)
            # part.add_header('Content-Disposition', "attachment", filename=fname+'.docx')  # fpath -> fname
            # msg.attach(part)

            # 메일 발송 요청
            s.sendmail(me, you, msg.as_string())

        # 전송완료 보고 / 서버 종료
        print('메일 전송이 완료되었습니다.')
        s.quit()
        return render(request, 'mail.html')

if __name__ == '__main__':
    mail_sender()
