# Persion AI Docent(Upstage X Seoul National University Mini Project)

## 1.Overview
Upsatge API를 활용하여 미술 작품에 대해 궁금한 점을 질의 응답할 수 있는 대화형 AI 구현

### Usage
.env 파일을 생성하여 UPSTAGE_API_KEY의 환경변수에 API키를 설정합니다.
```
UPSTAGE_API_KEY=up_71ks...
```
run.py 파이썬 파일을 실행해 gradio demo 챗봇을 실행할 수 있습니다.
```
python run.py
```
이후 검색을 통해서 설명을 듣고싶은 작품을 검색한 뒤 해당 작품에 대해서 질문할 수 있습니다.

![image](https://github.com/dudcjs2779/upstage-mini-project-art-chatbot/assets/42354230/d409b8a2-5e2d-4068-b3cf-2b17b09ee1af)

### 데모 시연
https://youtu.be/ddIHzZyz7vM

### etc
미술작품 데이터
- [국립현대미술관](https://www.mmca.go.kr/)의 50개의 미술작품 데이터를 수집
