import gradio as gr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage
from langchain_upstage import ChatUpstage

chat_with_history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
)
llm = ChatUpstage()
chain = chat_with_history_prompt | llm | StrOutputParser()

def chat(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    return chain.invoke({"message": message, "history": history_langchain_format})

# Create the Gradio interface
def move_to_chatbot(artwork_number):
    if artwork_number:  # 작품 번호가 입력되었을 때만 이동
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)

def go_back():
    return gr.update(visible=True), gr.update(visible=False)

with gr.Blocks() as demo:
    # 첫 번째 페이지: 작품 번호 입력
    with gr.Column(visible=True) as input_page:
        artwork_input = gr.Textbox(label="Enter Artwork Number")
        submit_btn = gr.Button("Submit")
    
    # 두 번째 페이지: 챗봇 UI
    with gr.Column(visible=False) as chatbot_page:
        chatbot = gr.ChatInterface(
            chat,
            examples=[
                "How to eat healthy?",
                "Best Places in Korea",
                "How to make a chatbot?",
            ],
            title="Solar Chatbot",
            description="Upstage Solar Chatbot",
        )
        chatbot.chatbot.height = 300
        back_btn = gr.Button("Back")

    # 작품 번호 제출 시 페이지 전환
    submit_btn.click(
        fn=move_to_chatbot, 
        inputs=artwork_input, 
        outputs=[input_page, chatbot_page]
    )

    # 뒤로 가기 버튼 클릭 시 첫 페이지로 전환
    back_btn.click(
        fn=go_back,
        inputs=[],
        outputs=[input_page, chatbot_page]
    )

# 인터페이스 실행
demo.launch()