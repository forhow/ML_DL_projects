from django.db import models


'''
    키워드에 대한 모델 생성
'''
class Keyword(models.Model):
    keyword = models.CharField(max_length=30)

    def __str__(self):
        return self.keyword


'''
    검색결과에 대한 모델 생성
'''
class Result(models.Model):
    keyword = models.ForeignKey(Keyword, on_delete=models.CASCADE)
    news = models.TextField()
    title = models.TextField()
    date = models.TextField()
    link = models.URLField()
    summary = models.TextField()
