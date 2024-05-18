import gradio as gr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage
from langchain_upstage import ChatUpstage
from tools import similar_art_search, chat_with_explain, normal_chat, wiki_search
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from embedding import prepare_embed

llm = ChatUpstage(streaming=True)
tools = [similar_art_search, chat_with_explain, normal_chat, wiki_search]
llm_with_tools = llm.bind_tools(tools)

def call_tool_func(tool_call):
    tool_name = tool_call["name"].lower()
    if tool_name not in globals():
        print("Tool not found", tool_name)
        return None
    selected_tool = globals()[tool_name]
    return selected_tool.invoke(tool_call["args"]), tool_name


# chain = prompt_template | llm | StrOutputParser()
df= pd.read_csv('arts02.csv')

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
    if tool_name == "similar_art_search":
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
        return None
    elif tool_name == "normal_chat":
        return None
    elif tool_name == "wiki_search":
        prompt =f"""
            당신은 미술 작품에 대한 해설사입니다. 당신의 역할은 미술 작품에 관심이 있는 상대방에게 미술에 관한 정보를 친절하게 설명해주고 알려주는 역할입니다.
            당신은 유저가 요청한 작품에 대해서 위키피디아에서 관련된 정보를 찾아서 알려주어야 합니다.
            아래는 질문과 관련된 작품설명입니다. 작품 설명을 기반으로 친절하게 설명해주세요.
            ---
            질문: {question}
            ---
            작품설명: {context}
            """
        return prompt

def chat(message, history, artwork_number):
    description = df[df["번호"]==int(artwork_number)]["작품 설명"].values[0]
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

def move_to_chatbot(artwork_number):
    if artwork_number:  # 작품 번호가 입력되었을 때만 이동
        return gr.update(visible=False), gr.update(visible=True), artwork_number
    else:
        return gr.update(visible=True), gr.update(visible=False), None

def go_back():
    return gr.update(visible=True), gr.update(visible=False)

def update_chatbot_title(artwork_number):
    return f"Solar Chatbot - Artwork Number: {artwork_number}"

with gr.Blocks() as demo:
    # 첫 번째 페이지: 작품 번호 입력
    with gr.Column(visible=True) as input_page:
        artwork_input = gr.Textbox(label="Enter Artwork Number")
        submit_btn = gr.Button("Submit")
    # 두 번째 페이지: 챗봇 UI
    with gr.Row(visible=False) as chatbot_page:
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
                additional_inputs=artwork_input
            )
            chatbot.chatbot.height = 300
            back_btn = gr.Button("Back")
            
        with gr.Column():
            image = gr.Image(f"data/{artwork_input.value}.jpg", label="Example Image")

    submit_btn.click(
        fn=move_to_chatbot, 
        inputs=artwork_input, 
        outputs=[input_page, chatbot_page, gr.State()]
    )

    back_btn.click(
        fn=go_back,
        inputs=[], 
        outputs=[input_page, chatbot_page, gr.State()]
    )

# 인터페이스 실행
demo.launch()