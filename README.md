# 얼굴 합성 포토샵 프로그램


**첫번째 파일**의 얼굴을 인식하여 **눈 , 코, 입**의 이미지를 images 파일에 저장하고, 

**두번째 파일**의 얼굴을 인식하여 **눈 , 코, 입**의 부분에 저장된 이미지를 배치시키고,

가우시간 블러, 컬러매치, 사이즈 조절, 블렌딩을 이용하여 얼굴합성을 시킨다.

얼굴인식은 Haar Cascades인 OpenCV GitHub에서 제공하는 사전 학습된 모델로 특정 객체(얼굴, 눈, 코, 입)를 인식하여 검출하였다.
해당 사이트 [haarcascades 깃허브](https://github.com/opencv/opencv/tree/master/data/haarcascades) 에서 제공받았다.

main함수를 실행하면 다음과 같은 창이 뜬다.


![image](https://github.com/kohjun/compositing-face/assets/82298792/e00f6406-7119-43d1-adc2-06020f12299f)


tkinter 창에서는 먼저 첫번째 이미지를 업로드를 눌러 jpg또는png 파일을 업로드한다.
성공적으로 대상을 인식하여 저장하였다면


![image](https://github.com/kohjun/compositing-face/assets/82298792/6708a77d-93a1-413b-aba8-d97568fffbea)

알림창이 떠서 눈, 코 ,입의 이미지가 images파일에 저장되었음을 알린다.

**[주의]**
Upload First Image와 Upload Second Image에 얼굴사진 샘플이 있으므로 업로드 하기 위해서는 해당 파일에 있는 이미지들을 바탕화면이나
다운로드 파일에 옮겨야한다. Compositing_Face 파일 안에 있는 이미지 파일을 업로드할 경우 열린파일이라서 업로드에 오류가 난다.

오류가 나지 않고  성공적으로 업로드가 되었다면

그후에 두번째 이미지 파일을 업로드한다.
성공적으로 업로드에 성공하면 곧바로 전 후를 비교하여 두번째 이미지에 첫번째 이미지에서 저장된 눈, 코 , 입을 배치시켜서 합성된 얼굴이 나온다.


예시1)

![image](https://github.com/kohjun/compositing-face/assets/82298792/a63b7b5b-8849-4cda-84e9-5f7a2fa2f829)

예시2)

![image](https://github.com/kohjun/compositing-face/assets/82298792/cb90299c-c4a4-4faa-99f8-cd57e9e84ac6)

예시3)
![image](https://github.com/kohjun/Face-Overlay-Photoshop/assets/82298792/c8378bfa-1a22-4416-bda5-80f8258adce4)


여러개의 사진을 통해서 다양한 인물의 합성 얼굴이 나온다.

*추가로*
여러 사진들을 통해 합성을 시도해본 결과 첫번째 이미지 파일은 kimtaehee.jpg 이미지가 가장 인식이 잘 되었다.
정면사진이 아닌 측면이나 눈, 코, 입의 경계가 모호한 사진들은 대상의 객체를 잘 인식하지 못하는 경우도 발생하였다.


밝기와, 명암, 채도를 조절해서 포토샵도 가능하다.

그 후에 완성된 사진은 저장하기를 눌러 images/outputs 파일에 저장된다.


face_compositor.py에는 

1. cv2.CascadeClassifier 얼굴의 눈, 코, 입을 검출하기 위해 Haar Cascade 분류기를 사용
2. cv2.cvtColor: 이미지를 그레이스케일로 변환
3. cv2.GaussianBlur: 얼굴의 특정 부분을 블러 처리
4. cv2.resize: 다른 이미지에서 추출한 얼굴 부분을 원본 이미지의 얼굴 부분 크기에 맞게 조정
5. cv2.imread 및 cv2.imwrite: 이미지를 불러오고 저장하는 데 사용
6. cv2.meanStdDev: 이미지의 평균 및 표준 편차를 계산하여 색상 매칭에 사용


위 기술 스택들은 이미지 처리에 필요한OpenCV에서 제공하는 다양한 기능들을 통해 얼굴합성에 용이하도록 도와주었다.



