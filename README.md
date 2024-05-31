# 얼굴합성 프로그램



**첫번째 파일**의 얼굴을 인식하여 눈 , 코, 입의 이미지를 images 파일에 저장하고, 

**두번째 파일**의 얼굴을 인식하여 눈 , 코, 입의 부분에 저장된 이미지를 배치시키고,

가우시간 블러, 컬러매치, 사이즈 조절, 블렌딩을 이용하여 얼굴합성을 시킨다.

얼굴인식은 Haar Cascades인 OpenCV GitHub에서 제공하는 사전 학습된 모델로 특정을 객체(얼굴, 눈, 코, 입)를 인식하여 검출하였다.
해당 사이트 [haarcascades 깃허브](https://github.com/opencv/opencv/tree/master/data/haarcascades) 에서 제공받았다.



main함수를 실행하면 다음과 같은 창이 뜬다.


![image](https://github.com/kohjun/compositing-face/assets/82298792/c2bf35be-0f0c-4ba8-be63-57f8106b49f9)


tkinter 창에서는 먼저 첫번째 이미지를 업로드를 눌러 jpg또는png 파일을 업로드한다.
성공적으로 대상을 인식하여 저장하였다면


![image](https://github.com/kohjun/compositing-face/assets/82298792/ca79b677-926d-4052-bb3b-4b33f71722ff)

Info 창이 떠서 눈, 코 ,입의 이미지가 images파일에 저장되었음을 알린다.


그후에 두번째 이미지를 업로드하면
전 후를 비교하여 두번째 이미지에 첫번째 이미지에서 저장된 눈, 코 , 입을 배치시켜서 합성된 얼굴이 나온다.


예시1)

![image](https://github.com/kohjun/compositing-face/assets/82298792/59b08558-9079-47c2-b095-31ccb791e918)

예시2)

![image](https://github.com/kohjun/compositing-face/assets/82298792/7d3b30d6-2092-4084-a1c4-2d56e1bbdded)



여러개의 사진을 통해서 다양한 인물의 합성 얼굴이 나온다.


face_compositor.py에는 

1. cv2.CascadeClassifier 얼굴의 눈, 코, 입을 검출하기 위해 Haar Cascade 분류기를 사용
2. cv2.cvtColor: 이미지를 그레이스케일로 변환
3. cv2.GaussianBlur: 얼굴의 특정 부분을 블러 처리
4. cv2.resize: 다른 이미지에서 추출한 얼굴 부분을 원본 이미지의 얼굴 부분 크기에 맞게 조정
5. cv2.imread 및 cv2.imwrite: 이미지를 불러오고 저장하는 데 사용
6. cv2.meanStdDev: 이미지의 평균 및 표준 편차를 계산하여 색상 매칭에 사용


위 기술 스택들은 이미지 처리에 필요한OpenCV에서 제공하는 다양한 기능들을 통해 얼굴합성에 용이하도록 도와주었다.



