import gradio as gr
import pandas as pd
import embedding
import json
import os
import random

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage
from langchain_upstage import ChatUpstage
from tools import similar_art_search, chat_with_explain, normal_chat, wiki_search
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatUpstage(streaming=True)
tools = [similar_art_search, chat_with_explain, normal_chat, wiki_search]
llm_with_tools = llm.bind_tools(tools)

df = pd.read_csv("data/arts02.csv")
searching = embedding.Search(df)
now_art = {}

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
        images_path = os.path.join("data", str(search_dict[item]['index']) + ".jpg") 
        global now_art
        now_art = search_dict[item]
        return images_path
    else:
        return None

def call_tool_func(tool_call):
    tool_name = tool_call["name"].lower()
    if tool_name not in globals():
        print("Tool not found", tool_name)
        return None
    selected_tool = globals()[tool_name]
    return selected_tool.invoke(tool_call["args"]), tool_name

def tool_rag(question, history):
    tool_calls = llm_with_tools.invoke(question).tool_calls
    if not tool_calls:
        return None, None
    context = ""
    for tool_call in tool_calls:
        tool_output = call_tool_func(tool_call)
        context, tool_name = tool_output
        context += str(context).strip()
        tool_name = str(tool_name)
    print(tool_name)
    if tool_name == "similar_art_search":
        # 비슷한 작품 검색
        docs = searching.search(f"{question}\n" + now_art['art_description'], searching.retriever_art)
        ran = random.randint(1, len(docs)-1)
        context = df[df["번호"]==docs[ran].metadata['index']]['작품 설명'].values[0]
        
        prompt = f"""
            당신은 미술 작품에 대한 해설사입니다. 당신의 역할은 미술 작품에 관심이 있는 상대방에게 미술에 관한 정보를 친절하게 설명해주고 알려주는 역할입니다.
            당신은 유저가 요청한 비슷한 작품에 대해서 설명해주어야 합니다. 
            ---
            질문: {question}
            ---
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

def chat(message, history):
    global now_art
    description = df[df["번호"]==now_art['index']]["작품 설명"].values[0]
    now_art['full_description'] = description
    now_art['art_description'] = df[df["번호"]==now_art['index']]["art_description"].values[0]
    
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
    output = tool_rag(message, history)
    if output:
        generator = chain.stream({"message": output, "history": history_langchain_format})
    else:
        generator = chain.stream({"message": message, "history": history_langchain_format})
    assistant = ""
    for gen in generator:
        assistant += gen
        yield assistant


with gr.Blocks(title="AI Docent Chatbot") as demo:
    gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Search Art</h1>")
    search_art_tb = gr.Textbox(label="Input query you want to search")
    search_dropdown = gr.Dropdown([], value='', label="Search Result", info="Art search results will be added here", interactive=True, filterable=False)
    search_btn = gr.Button("Search")
    search_to_meta = gr.Label(visible=False)
    
    with gr.Row(visible=True) as chatbot_page:
        with gr.Column():
            state = gr.State()
            chatbot = gr.ChatInterface(
                chat,
                # examples=[
                #     "How to eat healthy?",
                #     "Best Places in Korea",
                #     "How to make a chatbot?",
                # ],
                title="Solar Chatbot",
                description="Upstage Solar Chatbot",
                # additional_inputs=[search_dropdown, search_to_meta]
            )
            chatbot.chatbot.height = 300
        with gr.Column():
            art_image = gr.Image(value=None, label="Art Image")
    
    
    search_btn.click(
        fn=search_art, 
        inputs=search_art_tb, 
        outputs=[search_dropdown, search_to_meta]
    )
    
    search_dropdown.change(fn=dropdown_change, inputs=[search_dropdown, search_to_meta], outputs=art_image)

demo.launch()