from langchain_core.tools import tool
from embedding import prepare_embed
import wikipediaapi
from langchain_upstage import ChatUpstage


retriever = prepare_embed()

@tool
def similar_art_search(query: str) -> str:
    """비슷한 작품을 검색해서 추천해줍니다.
    """
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
    return "CALL EXPLAIN"

@tool
def normal_chat(query: str) -> str:
    """일반적인 대화를 수행합니다. 사용자가 미술 작품에 대한 검색을 요구하거나 감상을 제시하지 않았을 때 사용합니다.
    """
    return "CALL NORMAL CHAT"

@tool
def wiki_search(query: str) -> str:
    """위키피디아에서 관련된 키워드 관련 페이지를 불러옵니다.
    질문은 아트와 전시회 주제를 다루고 있습니다.
    """
    from ast import literal_eval
    llm = ChatUpstage()
    output = llm.invoke("아래는 유저가 검색을 하고 싶어하는 내용입니다. 아래 검색 내용에서 핵심 검색 키워드 1개만 추출해서 {'키워드' : 키워드 내용'}과 같은 형태로 출력해주세요. 키워드는 위키피디아에 있는 엔티티여야 합니다.\n검색 내용: {query}\n").content.strip()
    try:
        output = literal_eval(output)
    except:
        output = llm.invoke("아래는 유저가 검색을 하고 싶어하는 내용입니다. 아래 검색 내용에서 핵심 검색 키워드 1개만 추출해서 {'키워드' : 키워드 내용'}과 같은 형태로 출력해주세요. 키워드는 위키피디아에 있는 엔티티여야 합니다.\n검색 내용: {query}\n").content.strip()
        output = literal_eval(output)
    query = output['키워드'].strip()
    wiki = wikipediaapi.Wikipedia(
        language='ko',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent='my-custom-user-agent')
    
    wiki_page = wiki.page(title=query).summary
    summary = wiki_page.summary
    if wiki_page.exists() == False:
        return None
    else:
        return summary
