import gradio as gr
import pandas as pd
import embedding
import json
import os
import random
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage
from langchain_upstage import ChatUpstage
from tools import similar_art_search, chat_with_explain, normal_chat, wiki_search
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatUpstage(streaming=True)
tools = [similar_art_search, chat_with_explain, normal_chat, wiki_search]
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
        context, tool_name = tool_output
        context += str(context).strip()
        tool_name = str(tool_name)
    print(tool_name)
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
        ##
        질문: {question}
        
        ##
        비슷한 작품: {context}
        """
        return prompt
    elif tool_name == "chat_with_explain":
        prompt = f"""
당신은 미술 작품에 대한 감상을 듣고 자신의 생각을 얘기한 뒤 평가해야 합니다. 당신은 주관적인 감상을 경청하고 공감해야합니다. 또한 제공된 평가 기준에 따라 나의 감상에 대한 구체적인 피드백을 제공합니다. 
## 평가 기준:
1. 진솔성과 솔직함: 자신의 감정과 생각을 솔직하게 표현하는가?
2. 감정 표현의 풍부함: 다양한 감정을 풍부하게 표현하는가?
3. 독창적인 관점: 작품에 대한 독창적인 시각과 해석을 제시하는가?
4. 공감과 소통: 자신의 감상을 통해 다른 사람의 공감을 이끌어내는가?. 당신의 역할은 미술 작품에 관심이 있는 상대방에게 미술에 관한 정보를 친절하게 설명해주고 알려주는 역할입니다.
---
나의 감상: {question}\n
"""
        return prompt
    elif tool_name == "normal_chat":
        return None
    elif tool_name == "wiki_search":
        prompt =f"""
            당신은 미술 작품에 대한 해설사입니다. 당신의 역할은 미술 작품에 관심이 있는 상대방에게 미술에 관한 정보를 친절하게 설명해주고 알려주는 역할입니다.
            당신은 유저가 요청한 작품에 대해서 위키피디아에서 관련된 정보를 찾아서 알려주어야 합니다.
            아래는 질문과 관련된 검색결과입니다. 검색 결과를 기반으로 친절하게 설명해주세요.
            만약 위키피디아에서 정보를 가져오지 못했을 경우에는 검색에 실패하였다고 알려주세요.
            ---
            질문: {question}
            ---
            검색결과: {context}
            """
        return prompt

def chat(message, history, cur_art):
    cur_art = json.loads(cur_art)
    description = cur_art['작품 설명']

    chat_with_history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"당신은 미술 작품에 대한 해설사입니다. 당신의 역할은 미술 작품에 관심이 있는 상대방에게 미술에 관한 정보를 친절하게 설명해주고 알려주는 역할입니다. 모든 답변을 한국어로 해야합니다. 해당 작품에 대한 정보는 다음과 같습니다.\n\n작품 정보 : {description}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
    )
    chain = chat_with_history_prompt | llm | StrOutputParser()
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    # breakpoint()
    print(message)
    output = tool_rag(message, history, cur_art)
    print(output)
    print("history\n", history_langchain_format)

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

with gr.Blocks(title="AI Docent Chatbot", css=css) as demo:
    # 전시회 검색 UI
    with gr.Tab("전시 검색"):
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>전시 검색</h1>")
        ehb_search_tb = gr.Textbox(label="Query", info="전시회를 검색합니다.")
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
        ebh_art_gallery = gr.Gallery([], columns=6, height= 250, min_width=250, allow_preview=False, interactive=False)
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
                    title="Solar Chatbot",
                    description="Upstage Solar Chatbot",
                    autofocus=False
                )
                chatbot.chatbot.height = 600
            with gr.Column():
                art_image = gr.Image(value=None, label="Art Image", scale=1)

        
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