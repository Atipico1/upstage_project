import gradio as gr
import pandas as pd
import embedding
import json
import os
import random
from dotenv import load_dotenv
from PIL import Image, ImageOps
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage
from langchain_upstage import ChatUpstage
from tools import similar_art_search, qa_with_explain, empathize_with_user, normal_chat, wiki_search, archiving
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatUpstage(streaming=True)
tools = [similar_art_search, qa_with_explain, empathize_with_user, normal_chat, wiki_search, archiving]
llm_with_tools = llm.bind_tools(tools)

# 데이터 불러오기
art_df = pd.read_csv("data/arts_all02.csv")
ehb_df = pd.read_csv("data/exhibitions.csv")

ehb_data = ehb_df.to_json(force_ascii=False, orient="records")
ehb_data = json.loads(ehb_data)
ehb_image_path = "data/ehb_images"

# DB 및 Retrieval 불러오기
searching = embedding.Search(art_df, ehb_df)

def search_art(query):
    results = searching.search(query, searching.retriever_full)
    search_list = []
    serarch_dict = {}
    for result in results:
        element = result.metadata["author"] +": " + result.metadata["title"]
        search_list.append(element)
        serarch_dict[element] = result.metadata
    # print(serarch_dict)
    return gr.update(choices=search_list, value=None), json.dumps(serarch_dict, ensure_ascii=False)

def dropdown_change(item, search_dict):
    if item:
        search_dict = json.loads(search_dict)
        images_path = os.path.join("data/art_images", str(search_dict[item]['index']) + ".jpg") 
        search_art = art_df[art_df['번호'] == search_dict[item]['index']].iloc[0]
        return images_path, search_art.to_json(force_ascii=False), 
    else:
        return None, None

def call_tool_func(tool_call, question):
    tool_name = tool_call["name"].lower()
    if tool_name not in globals():
        print("Tool not found", tool_name)
        return None
    selected_tool = globals()[tool_name]
    print(tool_call)

    # query를 제대로 생성하지 못해 발생하는 오류방지
    if "query" not in tool_call["args"]:
        tool_call["args"] = {"query": question}
        print("query is empty", tool_call)

    return selected_tool.invoke(tool_call["args"]), tool_name

def tool_rag(question, history, cur_art):
    tool_calls = llm_with_tools.invoke(question).tool_calls
    if not tool_calls:
        return None, None
    context = ""
    for tool_call in tool_calls:
        tool_output = call_tool_func(tool_call, question)
        print(tool_output)
        context, tool_name = tool_output
        context += str(context).strip()
        tool_name = str(tool_name)
    print(tool_name)

    ## ==== similar_art_search ====
    if tool_name == "similar_art_search":
        # 비슷한 작품 검색
        docs = searching.search(f"{question}\n" + cur_art['art_description'], searching.retriever_art)
        ran = random.randint(1, len(docs)-1)
        context = art_df[art_df["번호"]==docs[ran].metadata['index']]['작품 설명'].values[0]
        
        prompt = f"""
## Role: 마술 작품 추천 전문가

## Instruction
- 사용자의 질문에 맞게 비슷한 작품을 소개해줘.
- 제시된 "비슷한 작품"을 기반으로 작품을 추천해줘.
- 이전 대화기록을 기반으로 추천하는 이유를 설명해줘.
## 질문
{question}

## 비슷한 작품
{context}
"""
        return prompt
    
    ##  ==== qa_with_explain ====
    elif tool_name == "qa_with_explain":
        prompt = f"""
## Role: 미술 작품 전문 해설가

## Instruction
- 미술 작품 전문 해설가로서 주어진 ## 미술 작품 정보 ## 를 읽고 상대방에게 질문에 답변합니다.
- 작품에 대한 설명을 요구할 경우 전반적인 작품에 대한 정보를 알기 쉽게 풀어서 전달합니다.
- 질문이 구체적인 경우 미술 작품의 정보를 참고해 간단하고 명료하게 대답합니다.
- 친절하고 상냥하게 답변합니다.
- 이전 대화기록을 참고하여 답변합니다.
- 한국어로 답변합니다.
- {cur_art['작품명']}에 대한 정보를 바탕으로 답변해야 합니다.

## 질문
{question}
"""
        return prompt
    
    ## ==== empathize_with_user ====
    elif tool_name == "empathize_with_user":
        prompt = f"""
## Role: 미술 작품 감상에 공감해주는 친구

## Instruction
- 사용자의 미술 작품에 대한 감상에 공감하고 감정을 공유합니다.
- 사용자의 감상에 관심을 보이고 더 깊은 공유를 위해 질문을 덧붙입니다.
- 사용자의 감상이 실제 작품에 대한 설명과 일치할 경우 그 근거를 들어 칭찬합니다.
- 친절하고 상냥하게 답변하되, 안녕하세요, 감사합니다, 물론이죠와 같은 인사말은 사용하지 않습니다.
- 이전 대화기록을 참고하여 답변합니다.
- 한국어로 답변합니다.

## 사용자의 감상
{question}
"""

        return prompt
    
    ## ==== normal_chat ====
    elif tool_name == "normal_chat":
        return None
    
    ## ==== wiki_search ====
    elif tool_name == "wiki_search":
        prompt = f"""
## Role: 미술 작품 검색기

## Instruction
- 미술 작품에 대한 정보를 위키피디아에서 검색하여 제공합니다.
- 검색한 정보를 기반으로 미술 작품에 대해 설명합니다.
- 만약 위키피디아에서 정보를 가져오지 못했을 경우에는 검색에 실패하였다고 알려주세요.
- 이전 대화기록을 참고하여 답변합니다.
- 한국어로 답변합니다.

## 질문
{question}

## 검색결과
{context}
"""
        return prompt
    
    elif tool_name == "archiving":
        prompt = f"""
## Role: 전시회 후기 작성

## Instruction
- 사용자와 나눈 대화를 SNS에 공유하기 위한 전시회 관람 후기 형태로 작성합니다.
- 관람 후기는 1인칭으로 작성되어야 하며 아래와 같은 형식으로 작성합니다
    1. 작품명
    2. 설명
    3. 감상 (사용자의 감상을 그대로 사용하되 첫글자는 감정을 이모티콘으로 표현)
- 사용자의 특별한 요청이 있다면 그에 맞게 작성합니다.
- 여러 개의 작품에 대한 감상이 있을 경우, 각 작품에 대한 감상을 모두 포함하여 작성하고, 작품 간 줄바꿈을 활용하여 구분합니다.
- 후기는 SNS에 올릴 수 있도록 간결하게 작성하고, "~했음"과 같은 말투를 사용해야 합니다.
- 마크다운 문법을 사용하면 안되고 Plain text만 사용해야 합니다.
- 이모티콘으로 대체가 가능한 텍스트는 이모티콘을 적극적으로 사용해야 하고 적절한 줄바꿈과 글머리 기호가 사용되어야 합니다.
- 마지막에는 전시회에 대한 전반적인 감상을 "한줄평:"에 1~2줄로 요약합니다.

## 사용자의 요청
{question}

## 전시회 후기:
"""
        return prompt

def chat(message, history, cur_art):
    cur_art = json.loads(cur_art)

    basic_prompt =f"""
## Role: 미술 작품 해설가

## Instruction
- 주어진 미술 작품 정보를 기반으로 사용자와 대화합니다.

## 미술 작품 정보
작가: {cur_art['작가']}
작품명: {cur_art['작품명']}
제작연도: {cur_art['제작연도']}
작품 재료: {cur_art['재료']}
작품 규격: {cur_art['규격']}
작품 설명: {cur_art['작품 설명']}
"""

    chat_with_history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", basic_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
    )
    chain = chat_with_history_prompt | llm | StrOutputParser()
    history_langchain_format = [AIMessage(content=basic_prompt)]
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    output = tool_rag(message, history, cur_art)
    if output:
        generator = chain.stream({"message": output, "history": history_langchain_format})
    else:
        generator = chain.stream({"message": message, "history": history_langchain_format})
    assistant = ""
    for gen in generator:
        assistant += gen
        yield assistant

# 전시 검색
def ehb_search(query, *images):
    result = searching.bm_search(query, searching.retriever_ehb)
    ehb_list = []
    for i in range(len(images)):
        if i < len(result):
            img_path = os.path.join(ehb_image_path, str(result[i].metadata['index'])+".jpg")
            ehb_image = gr.Image(img_path, label=result[i].metadata["title"], width=300, height=300, show_label=False, show_download_button=False, container=False, interactive=False)
            ehb_title = gr.Markdown(f"<div class='ehb_label'>{result[i].metadata['title']}</div>")
            ehb_list.append(ehb_image)
            ehb_list.append(ehb_title)

        else:
            ehb_list.append(gr.Image(visible=False))
            ehb_list.append(gr.Image(visible=False))

    result_list = []
    for doc in result:
        result_list.append(doc.metadata)

    return json.dumps(result_list), *ehb_list

# 전시회 선택
def ehb_select(value, result, evt: gr.EventData):
    result = json.loads(result)
    index = os.path.splitext(evt.target.value['orig_name'])[0]
    sel_ehb = result[int(index)-1]
    indice = eval(sel_ehb['art_list'])
    ehb_arts = art_df[art_df['번호'].isin(indice)]

    ehb_arts_img_paths = tuple(map(lambda x: os.path.join("data/art_images", str(x)+".jpg"), indice))
    gal_imgs = list(zip(ehb_arts_img_paths, ehb_arts['작품명']))

    return gr.Gallery(gal_imgs, columns=6, height= 250, min_width=250, allow_preview=False, interactive=False), json.dumps(sel_ehb, ensure_ascii=False)

# 전시회 작품 선택
def ehb_art_on_select(value, cur_ehb, evt: gr.SelectData):
    cur_ehb = json.loads(cur_ehb)
    indice = eval(cur_ehb['art_list'])
    art_idx = indice[evt.index]
    sel_art = art_df[art_df["번호"] == art_idx].iloc[0]
    img_path = os.path.join("data/art_images", str(sel_art['번호']) + ".jpg")

    return sel_art.to_json(force_ascii=False), gr.Image(img_path), gr.update(visible=True)

css = """
#ehb_image {background-color: #FFFFFF; align-items: center; width: 250px;}
#ehb_image img {width: 300px; height: 300px; padding: 10px}
#ehb_image .ehb_label {width: 250px; text-align: left; padding-left: 45px; font-size: 18px; font-weight: bold}
#chat_img img {width: 600px; height: 600px; align-items: center;}
#chat_img Column {align-items: center;}
"""

with gr.Blocks(title="DocentAI", css=css, theme=gr.themes.Soft()) as demo:
    # 전시회 검색 UI
    with gr.Tab("전시 검색"):
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>전시 검색</h1>")
        ehb_search_tb = gr.Textbox(label="전시회 검색", info="전시회 이름을 입력해주세요.")
        ehb_search_btn = gr.Button("Search")
        
        # 전시검색 결과 초기화
        result = ehb_df.drop(columns="이미지")
        result.columns = ['index', 'title', 'art_list']
        result = result.to_json(orient='records', force_ascii=False)
        ehb_search_result = gr.State(result)

        # 전시 목록
        gr.Markdown("<h2 style='text-align: left; margin-bottom: 1rem'>전시회</h2>")
        ehb_list = []
        with gr.Group():
            with gr.Row():
                for item in ehb_data:
                    img_path = os.path.join(ehb_image_path, str(item['번호'])+".jpg")
                    with gr.Column(elem_id="ehb_image", min_width=250):
                        ehb_image = gr.Image(img_path, label=item["전시"], width=300, height=300, show_label=False, show_download_button=False, container=False, interactive=False)
                        ehb_title = gr.Markdown(f"<div class='ehb_label'>{item['전시']}</div>")
                        ehb_list.append(ehb_image)
                        ehb_list.append(ehb_title)
                        
        # 선택한 전시회 작품들
        gr.Markdown("<h2 style='text-align: left; margin-bottom: 1rem'>전시 작품</h2>")
        ebh_art_gallery = gr.Gallery([], columns=6, height= 250, min_width=250, allow_preview=False, show_download_button=False, interactive=False)
        cur_ehb_tb = gr.Textbox(label="cur_ehb_tb" , visible=False)
        cur_art_tb = gr.Textbox(label="cur_art_tb" , visible=False)
        

        # 챗봇
        with gr.Row(visible=False) as chatbot_ehb:
            with gr.Column(scale=1.2):
                state = gr.State()
                chatbot = gr.ChatInterface(
                    chat,
                    # examples=[
                    #     "How to eat healthy?",
                    #     "Best Places in Korea",
                    #     "How to make a chatbot?",
                    # ],
                    additional_inputs=[cur_art_tb],
                    title="AI Docent",
                    description="작품에 대해 궁금한게 있다면 말씀해주세요!",
                    autofocus=False
                )
                chatbot.chatbot.height = 600
            with gr.Column():
                art_image = gr.Image(value=None, label="작품 이미지", scale=1)

        
        ehb_search_btn.click(
            fn=ehb_search, 
            inputs=[ehb_search_tb] + ehb_list, 
            outputs= [ehb_search_result, *ehb_list]
        )

        for i in range(0, len(ehb_list), 2):
            ehb_list[i].select(
                fn=ehb_select,
                inputs=[ehb_list[i], ehb_search_result],
                outputs=[ebh_art_gallery, cur_ehb_tb]
            )

        ebh_art_gallery.select(
            fn=ehb_art_on_select, 
            inputs=[ebh_art_gallery, cur_ehb_tb], 
            outputs=[cur_art_tb, art_image, chatbot_ehb], 
            trigger_mode="once"
        )
        

    with gr.Tab("전체 작품 검색"):
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>전체 작품 검색</h1>")
        search_art_tb = gr.Textbox(label="Query", info="작품에 대한 정보를 입력해주세요.")
        search_dropdown = gr.Dropdown([], value='', label="Search Result", info="Art search results will be added here", interactive=True, filterable=False)
        search_btn = gr.Button("Search")
        search_to_meta = gr.Label(visible=False)
        cur_search_art_tb = gr.Textbox(label="search_art_tb" , visible=False)
        
        with gr.Row() as chatbot_art:
            with gr.Column(scale=1):
                state = gr.State()
                chatbot = gr.ChatInterface(
                    chat,
                    # examples=[
                    #     "How to eat healthy?",
                    #     "Best Places in Korea",
                    #     "How to make a chatbot?",
                    # ],
                    additional_inputs=[cur_search_art_tb],
                    title="Solar Chatbot",
                    description="Upstage Solar Chatbot",
                    autofocus=False
                )
                chatbot.chatbot.height = 600
            with gr.Column(scale=1):
                art_image = gr.Image(value=None, label="Art Image")
        
        search_btn.click(
            fn=search_art, 
            inputs=search_art_tb, 
            outputs=[search_dropdown, search_to_meta]
        )
        
        search_dropdown.change(fn=dropdown_change, 
                               inputs=[search_dropdown, search_to_meta], 
                               outputs=[art_image, cur_search_art_tb]
                               )

demo.launch()