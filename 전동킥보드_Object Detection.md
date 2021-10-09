# 전동킥보드_Object Detection 1차

### 1) Bing에서 **데이터 크롤링 - 100장  (keyword : 도보 킥보드)**

```python
# Bing에서 받은 데이터(테스트 데이터)
######################################################

import urllib.request # 웹 url을 파이썬이 인식 할 수 있게하는 패키지
from  bs4 import BeautifulSoup # html에서 데이터 검색을 용이하게 하는 패키지
from selenium import webdriver  
# selenium : 웹 애플리케이션의 테스트를 자동화하기 위한 프레임 워크 
# 손으로 클릭하면서 컴퓨터가 대신하면서 스크롤링하게 하는 패키지

from selenium.webdriver.common.keys import Keys
import time       # 중간중간 sleep 을 걸어야 해서 time 모듈 import

########################### url 받아오기 ###########################

# 웹브라우져로 크롬을 사용할거라서 크롬 드라이버를 다운받아 아래 파일경로의 위치에 둔다
# 팬텀 js로 하면 백그라운드로 실행할 수 있음
binary = 'c:\chromedriver/chromedriver.exe' 

# 브라우져를 인스턴스화
browser = webdriver.Chrome(binary)

# Bing의 이미지 검색 url 받아옴(아무것도 안 쳤을때의 url) 
browser.get("https://www.bing.com/images?FORM=Z9LH")
time.sleep(2)
# 구글의 이미지 검색에 해당하는 input 창의 id 가 '  ?  ' 임(검색창에 해당하는 html코드를 찾아서 elem 사용하도록 설정)
# input창 찾는 방법은 원노트에 있음

# elem = browser.find_elements_by_class_name('b_searchbox') # Tip : f12누른후 커서를 검색창에 올려두고 class이름이나 id 찾아보기.

elem = browser.find_element_by_xpath("//*[@class='sb_form_q']")  # 위의 코드대로 하거나 이렇게 하거나 둘 중 하나 select

########################### 검색어 입력 ###########################

# elem 이 input 창과 연결되어 스스로 햄버거를 검색
elem.send_keys("도보 킥보드") # 여기에 스크롤링하고싶은 검색어를 입력

# 웹에서의 submit 은 엔터의 역할을 함
elem.submit()

# 현재 결과 더보기는 구현 되어있지 않은상태 -> 구글의 경우 400개 image가 저장됨.

########################### 반복할 횟수 ###########################

# 스크롤을 내리려면 브라우져 이미지 검색결과 부분(바디부분)에 마우스 클릭 한번 하고 End키를 눌러야함
for i in range(1, 5): # 4번 스크롤 내려가게 구현된 상태 range(1,5)
    browser.find_element_by_xpath("//body").send_keys(Keys.END)
    time.sleep(10)          # END 키 누르고 내려가는데 시간이 걸려서 sleep 해줌 / 키보드 end키를 총 5번 누르는데 end1번누르고 10초 쉼

time.sleep(10)                      # 네트워크 느릴까봐 안정성 위해 sleep 해줌(이거 안하면 하얀색 이미지가 다운받아질 수 있음.)
html = browser.page_source         # 크롬브라우져에서 현재 불러온 소스 가져옴
soup = BeautifulSoup(html, "html.parser") # html 코드를 검색할 수 있도록 설정

########################### 그림파일 저장 ###########################

### 검색한 bing 이미지의 url을 따오는 코드 ###
def fetch_list_url():
    params = []
    imgList = soup.find_all("img", class_= "mimg")  # bing 이미지 url 이 있는 img 태그의 _img 클래스에 가서 (f12로 확인가능.)
    for im in imgList:
        try :
            params.append(im["src"])                   # params 리스트에 image url 을 담음.
        except KeyError:
            params.append(im["data-src"])
    return params

# except부분
# 이미지의 상세 url 의 값이 있는 src 가 없을 경우
# data-src 로 가져오시오 ~  

def fetch_detail_url():
    params = fetch_list_url()

    for idx,p in enumerate(params,1):
        # 다운받을 폴더경로 입력
        urllib.request.urlretrieve(p, "d:\\pre-winner\\kick\\" + str(idx) + ".jpg")  # 저장 폴더 위치 바꾸기

# enumerate 는 리스트의 모든 요소를 인덱스와 쌍으로 추출
# 하는 함수 . 숫자 1은 인덱스를 1부터 시작해라 ~

fetch_detail_url()

# 끝나면 브라우져 닫기
browser.quit()
```

### 2) 코랩에서 사물검출 수행

- classes : 1
- accuracy : 91.65 %
- max batch : class * 2000 = 2000
- filter : (4+1+classes)*3 = 18

```python
##### 1.GPU 확인 #####
!nvidia-smi -L

##### 2. 구글드라이브 연동 #####
from google.colab import drive
drive.mount('/content/drive')

##### 3. 파일압축 해제 #####
# /content/drive/MyDrive 밑에 yolo_custom_model_Training3라는 폴더를 생성
%cd /content/drive/MyDrive
!mkdir yolo_custom_model_Training3

# yolo_custom_model_Training3 밑에 custom_data.zip 파일을 올리고 잘 올라갔는지 확인
!ls '/content/drive/MyDrive/yolo_custom_model_Training3'

# yolo_custom_model_Training3 디렉토리로 이동하여 custom_data 폴더를 생성
%cd /content/drive/MyDrive/yolo_custom_model_Training3
!mkdir custom_data

# /content/drive/MyDrive/yolo_custom_model_Training3/custom_data 폴더 밑에 압축을 해제
!unzip '/content/drive/MyDrive/yolo_custom_model_Training3/custom_data.zip' -d '/content/drive/MyDrive/yolo_custom_model_Training3/custom_data'

# yolo_custom_model_Training3 폴더로 이동
%cd /content/drive/MyDrive/yolo_custom_model_Training3

```

```python

##### 4. 다크넷 다운로드 #####
# 다크넷(YOLO 구현하기위해 C로 만든 신경망) 다운로드 
!git clone 'https://github.com/AlexeyAB/darknet.git' '/content/drive/MyDrive/yolo_custom_model_Training3/darknet'

# 다운받은 다크넷이 있는 폴더로 이동
%cd /content/drive/MyDrive/yolo_custom_model_Training3/darknet

# Makefile 의 내용을 수정 (다크넷 환경설정 파일에 GPU 사용할 것( =1)이라고 설정)
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile

# Compile model (수정된 make 파일을 실행하여 다크넷 모델을 컴파일)
"""  take care do not disconnect : file directory may be interupted 
if your network down during compile, I recommend delete darknet folder and restart number 4(get AlexeyAB/darknet)"""
!make

# darknet 을 적용 (상위 디렉토리로 이동해서 다크넷 모델 세팅)
%cd ..
!darknet/darknet

```

```python

##### 5. 훈련/테스트데이터 분리 #####
# 다음의 github 주소에서 사물검출에 필요한 코드들을 다운로드
!git clone 'https://github.com/jakkcoder/training_yolo_custom_object_detection_files' '/content/drive/MyDrive/yolo_custom_model_Training3/training_yolo_custom_object_detection_files-main'

#  코드를 다운로드 받은 경로로 이동
%cd /content/drive/MyDrive/yolo_custom_model_Training3/training_yolo_custom_object_detection_files-main

# ls 를 수행하여 아래와 같이 4개의 파일이 존재하는지 확인
"""
creating-files-data-and-name.py   -> 200개 이미지를 훈련과 테스트로 분리
creating-train-and-test-txt-files.py  -> 200개 이미지를 훈련과 테스트로 분리
'Custom Object detection live video.ipynb'  -> 동영상 detection
Rename_files.ipynb  -> file이름 변경
"""
!ls

# copy creating-train-and-test-txt-files.py & creating-files-data-and-name.py
# 훈련과 테스트로 데이터를 분리하는 파이썬 파일2개를 우리가 만든 이미지와 라벨링된 텍스트가 있는 폴더로 복사
"""creating-train-and-test-txt-files.py >> create 'train.txt' & 'test.txt' files
   creating-files-data-and-name.py >> create label 'labelled_data.data' file
   <if you excute both .py files, you get mentioned files upper lines 2,3>"""

!cp creating-train-and-test-txt-files.py /content/drive/MyDrive/yolo_custom_model_Training3/custom_data
!cp creating-files-data-and-name.py /content/drive/MyDrive/yolo_custom_model_Training3/custom_data

# 학습할 이미지와 라벨링된 텍스트가 있는 폴더로 이동
%cd /content/drive/MyDrive/yolo_custom_model_Training3/custom_data

# change paths in both .py files (2개의 파이썬 코드의 내용을수정)
!sed -i '39 s@/home/my_name/Downloads/video-to-annotate@custom_data@' creating-train-and-test-txt-files.py
!sed -i '74 s@jpeg@jpg@' creating-train-and-test-txt-files.py
!sed -i '36 s@/home/my_name/Downloads/video-to-annotate@custom_data@' creating-files-data-and-name.py

# 상위 디렉토리로 이동
%cd ..

# excute .py file >> 'train.txt', 'test.txt'   
# 훈련데이터의 명단과 테스트데이터의 명단을 만드는 작업
!python custom_data/creating-train-and-test-txt-files.py

# excute .py file >> 'labelled_data.data'
# 정답데이터 생성
!python custom_data/creating-files-data-and-name.py

```

```python

##### 6. YOLO 환경설정 파일 세팅 #####
#구글에서 yolov4.conv.137로 검색을 하고 모델을 다운로드 받아 /content/drive/My Drive/yolo_custom_model_Training3 에 업로드하기

%cd /content/drive/MyDrive/yolo_custom_model_Training3

# create directory 'custom_weight'
# YOLO모델을 저장할 디렉토리 생성
!mkdir custom_weight

# move 'yolov4.conv.137' file to 'custom_weight' dir (YOLO모델을 custom_weight로 이동)
!mv yolov4.conv.137 custom_weight/

# 다크넷 모델의 환경설정 파일이 있는곳으로 이동
%cd /content/drive/MyDrive/yolo_custom_model_Training3/darknet/cfg

# copy yolov4.cfg file & rename & paste
# yolov4.cfg(욜로 환경설정 파일)을  yolov4_custom.cfg 로 복사
!cp yolov4.cfg yolov4_custom.cfg

# YOLO 다크넷 모델을 학습시킬때 사용하는 환경설정 파일을 변경 
# 이 작업을 수행하지 않으면 메모리 초과오류로 인해 학습이 안되는 경우가 종종 생기기 때문
# 학습할때 메모리 초과오류를 방지하려면 배치사이즈를 조정해야함

!sed -i '2 s@batch=64@batch=8@' yolov4_custom.cfg

!sed -i '7 s@width=608@width=416@' yolov4_custom.cfg
!sed -i '8 s@height=608@height=416@' yolov4_custom.cfg  

!sed -i '19 s@500500@2000@' yolov4_custom.cfg  # maxbatch=class*2000=2000   class=1
!sed -i '21 s@400000,450000@1600,1800@' yolov4_custom.cfg  # maxbatch*0.8=1600, maxbatch*0.9=1800

!sed -i '968 s@classes=80@classes=1@' yolov4_custom.cfg
!sed -i '1056 s@classes=80@classes=1@' yolov4_custom.cfg
!sed -i '1144 s@classes=80@classes=1@' yolov4_custom.cfg

!sed -i '961 s@filters=255@filters=18@' yolov4_custom.cfg  # filters=(4+1+classes)*3 = 18
!sed -i '1049 s@filters=255@filters=18@' yolov4_custom.cfg
!sed -i '1137 s@filters=255@filters=18@' yolov4_custom.cfg

# 학습될 가중치를 저장할 backup 이라는 폴더를 생성
%cd /content/drive/MyDrive/yolo_custom_model_Training3
!mkdir backup

```

```python
##### 7. 모델 훈련 #####
# 다크넷을 이용하여 학습 데이터들을 훈련 (1시간정도 소요)
!darknet/darknet detector train custom_data/labelled_data.data darknet/cfg/yolov4_custom.cfg custom_weight/yolov4.conv.137 -dont_show

# 결과를 시각화 할 함수를 생성
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  #plt.show('')

# 학습결과를 확인
imShow('chart.png')

%cd custom_data

!sed -i 's@custom_data@/content/drive/MyDrive/yolo_custom_model_Training3/custom_data@' test.txt
!sed -i 's@custom_data@/content/drive/MyDrive/yolo_custom_model_Training3/custom_data@' train.txt

!sed -i 's@custom_data@/content/drive/MyDrive/yolo_custom_model_Training3/custom_data@' labelled_data.data
!sed -i '5 s@.*@backup = /content/drive/MyDrive/yolo_custom_model_Training3/backup/@' labelled_data.data

!cat labelled_data.data

# 다크넷 디렉토리의 실행권한을 넣어줌
%cd /content/drive/My Drive/yolo_custom_model_Training3/darknet
!chmod +x ./darknet

#You can check the mAP for all the saved weights to see which gives the best results
# YOLO모델의 각 클래스별 성능 확인
!./darknet detector map /content/drive/MyDrive/yolo_custom_model_Training3/custom_data/labelled_data.data /content/drive/MyDrive/yolo_custom_model_Training3/darknet/cfg/yolov4_custom.cfg /content/drive/MyDrive/yolo_custom_model_Training3/backup/yolov4_custom_final.weights -points 0

```

```python

##### 8. 이미지에서 사물검출 테스트 #####

# 다크넷의 환경구성 파일인 cfg 디렉토리로 이동해서 테스트데이터 1장을 테스트해볼거라서 batch를 1로 지정
%cd /content/drive/MyDrive/yolo_custom_model_Training3/darknet/cfg
!sed -i 's/batch=64/batch=1/' yolov4_custom.cfg
!sed -i 's/subdivisions=16/subdivisions=1/' yolov4_custom.cfg
%cd ..

# 학습하지 않았던 새로운 이미지로 사물검출 수행
!./darknet detector test /content/drive/MyDrive/yolo_custom_model_Training3/custom_data/labelled_data.data /content/drive/MyDrive/yolo_custom_model_Training3/darknet/cfg/yolov4_custom.cfg /content/drive/MyDrive/yolo_custom_model_Training3/backup/yolov4_custom_final.weights /content/drive/MyDrive/yolo_custom_model_Training3/kick3.png -thresh 0.3
imShow('predictions.jpg')
```

![predictions.jpg](%E1%84%8C%E1%85%A5%E1%86%AB%E1%84%83%E1%85%A9%E1%86%BC%E1%84%8F%E1%85%B5%E1%86%A8%E1%84%87%E1%85%A9%E1%84%83%E1%85%B3_Object%20Detection%201%E1%84%8E%E1%85%A1%2027520e191234499f964a63b25a42ec9b/predictions.jpg)

![predictions (1).jpg](%E1%84%8C%E1%85%A5%E1%86%AB%E1%84%83%E1%85%A9%E1%86%BC%E1%84%8F%E1%85%B5%E1%86%A8%E1%84%87%E1%85%A9%E1%84%83%E1%85%B3_Object%20Detection%201%E1%84%8E%E1%85%A1%2027520e191234499f964a63b25a42ec9b/predictions_(1).jpg)

![predictions (2).jpg](%E1%84%8C%E1%85%A5%E1%86%AB%E1%84%83%E1%85%A9%E1%86%BC%E1%84%8F%E1%85%B5%E1%86%A8%E1%84%87%E1%85%A9%E1%84%83%E1%85%B3_Object%20Detection%201%E1%84%8E%E1%85%A1%2027520e191234499f964a63b25a42ec9b/predictions_(2).jpg)

```python
##### 9. 실시간 웹캠의 이미지로 테스트 #####

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))

!./darknet detector test /content/drive/MyDrive/yolo_custom_model_Training3/custom_data/labelled_data.data /content/drive/MyDrive/yolo_custom_model_Training3/darknet/cfg/yolov4_custom.cfg /content/drive/MyDrive/yolo_custom_model_Training3/backup/yolov4_custom_final.weights photo.jpg -thresh 0.5
imShow('predictions.jpg')
```

![predictions (3).jpg](%E1%84%8C%E1%85%A5%E1%86%AB%E1%84%83%E1%85%A9%E1%86%BC%E1%84%8F%E1%85%B5%E1%86%A8%E1%84%87%E1%85%A9%E1%84%83%E1%85%B3_Object%20Detection%201%E1%84%8E%E1%85%A1%2027520e191234499f964a63b25a42ec9b/predictions_(3).jpg)