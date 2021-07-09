'''
    AI History

    Confusion matrix

    Activation function
    Loss function
    Optimizer


    XOR 문제 해결 : 레이어(차원)를 늘려서 해결 : 너무 많으면 차원의 저주
'''

'''
    AI History
    
인공지능은 인간의 뇌를 모델링한 구조
- 앞단의 뉴런과 뒷단의 뉴런이 얼마나 강하게 연결되어 있는지에 따라
 신호의 강도가 달라짐  --> weight
- 뉴런자체의 민감도, 특성 --> bias
- 뉴런은 활성(반응,1)과 비활성화(무반응,0)으로 작동 --> activation function
 : input * weight + bias 값을 활성화되는 조건과 비교해서 활성화/비활성화 결정
 : 특정 기준을 두고 step function으로 구성한 경우 미분이 불가(첨점, 중단점이 있으면 미분불가)
 : 영향을 미치는 값이 크면 큰 가중치를, 적으면 적은 가중치를 적용해서 조정해야하나,
   미분이 불가하면 영향도를 측정할 수가 없음 - sigmoid를 발견
 : sigmoid는 일정 값 이상/이하에서 0/1을 반환하지만, 미분을 한 값이 0.1~0.3 값 이내로 범위가 작아짐
   > 레이어가 많아질 수록 미분값이 점점 작아져 gradient vanishing 문제 발생
 : 기울기 소실문제 해결하기 위해 Relu, tanh, leaky relu 등이 대두됨
 : 2진분류의 경우 맨 끝단은 sigmoid, 중간단계는 보통 relu를 사용 

- 파블로프의 개 : 훈련(학습)에 의해 특정 조건에 반응이 강해지도록 조정가능함을 적용
 : 특정 조건에 민감하게 반응하도록 가중치를 적용하도록 함

- 퍼셉트론 구성 : weight, bias, activation function 으로 구성
'''

'''
    Activation function
    
 - 2진분류 : sigmoid
 - 다중분류 : softmax
 - 실제값 : 활성화 함수 비적용 

'''

'''
    Confusion matrix
    
        - 정리자료 그림 참조
    
Type1 error : 실제 negative지만 positive를 예측하는 에러
Type2 error : 실제 Positive지만 Negative를 예측하는 에러

- 의학 등 분야의 민감한 경우 : 실제 음성(Negative)인데 양성(positive)으로 예측하는 경우 (FP; False Positive)를 주의해야 함
    > Type1 error 주의 ; 실제 음성 환자의 치료기회가 늦어짐 (반면, 양성인데 음성판정한 경우는 추가 검사로 확인가능)
    > Precision 높임 
- spam 메일의 경우 : 실제 필요한 메일(Positive)이지만 스팸(Negative)으로 분류하는 경우 (FN; False Negative)를 주의해야 함
    > Type 2 error 주의  ; 필요한 메일을 확인하지 못해 업무처리가 늦어짐 (반면, 스팸메일이 메일함에 있는 경우는 직접 스팸처리 가능함)
    > Sensitivity 높임
- 전체 case에 대한 정확도 : 정분류율 = 맞춘개수 / 전체개수
- F1 = 2 * 정밀도 + 재현율 / 정밀도 * 재현율
  > F1 score = binary accuracy


분류문제 - loss보다 accuracy가 더 중요
 > 2진분류에서는 bianry accuracy로 확인

'''

'''
    Loss function
    
MSE Mean Square Error : 평균 제곱오차
'''

'''
    Optimizer

참고 : https://choosunsick.github.io/post/optimizer_compare/
- compile 과정에서 사용, 가중치와 편향의 갱신 방법 지정
- 경사하강 알고리즘에서 기울기가 0인점이 2개인 경우 최저점을 찾을 수 없는 확률이 생김
- adam : 최저점을 찾아가는 성능면에서 좋음,
 > 특정한 경우 다른 optimizer를 사용하나 차후 경우에 대해 얘기함
- SGD : 통계적 경사하강, 한 개의 축 방향에서만 분석함

'''

'''
input dim : input 개수와 같게 설정
hidden layer개수 : 적절히 설정
'''