from langchain_core.tools import tool

@tool
def similar_art_search(query: str, vector_store) -> str:
    """비슷한 작품을 검색해서 추천해줍니다.
    """
    retriever = vector_store.get_retriever()
    result = retriever.invoke(query)[0].page_content
    return result

@tool
def chat_with_explain(query: str) -> str:
    """사용자의 작품에 대한 감상에 대해서 평가해줍니다.
    """
    prompt = """아래는 미술 작품에 대한 감상입니다. 아래 기준에 따라서 감상에 대한 의견을 제시해주세요.
    친구와 대화하듯이 자연스럽게 대화를 이어 나가야하고, 감상에 대한 평가는 기준에 따라서 해야합니다.
    기준 : {NORM}\n
    감상 : {USER}\n
    """