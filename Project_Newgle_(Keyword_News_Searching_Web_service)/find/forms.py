from django import forms
from find.models import Keyword, Result

class KeywordForm(forms.ModelForm):
    class Meta:
        model = Keyword
        fields = ['keyword']
        labels = {'keyword': '키워드',
                  }

# class ResultForm(forms.ModelForm):
#     class Meta:
#         model = Result
#         fields = ['news', 'title', 'date', 'link', 'summary']
#         labels = {'news' : '언론사',
#                   'title':'기사제목',
#                   'date' : '일자',
#                   'link' : '원문주소',
#                   'summary': '내용',
#                   }
class EmailForm(forms.Form):
    mail_sender = forms.EmailField()
