from langchain_core.tools import tool
import wikipediaapi
from langchain_upstage import ChatUpstage

@tool
def similar_art_search(query: str) -> str:
    """사용자와의 대화 기록을보고 사용자가 비슷한 작품의 추천을 원할시 호출합니다.
    """
    return "CALL SIMILAR ART"

@tool
def qa_with_explain(query: str) -> str:
    """주어진 미술 작품 정보를 기반으로 사용자의 질문에 답변합니다. 미술작품의 설명을 요구하거나 작품에 대한 구체적인 질문을 할 경우 호출합니다.
    """
    return "CALL QA"

@tool
def empathize_with_user(query: str) -> str:
    """사용자가 작품에 대한 감상을 이야기하면 공감해주고 의견을 공유합니다. 사용자가 작품에 대한 감상을 공유하고자 할때 호출합니다.
    """
    return "CALL EXPLAIN"

@tool
def normal_chat(query: str) -> str:
    """사용자가 미술 작품과 관련이 없는 이야기를 할 경우 평범하게 대화합니다.미술 작품과 관련이 없는 이야기를 하는 경우 호출합니다.
    """
    return "CALL NORMAL CHAT"

@tool
def wiki_search(query: str) -> str:
    """미술 작품에 대한 정보를 위키피디아에서 검색하여 제공합니다. 사용자가 미술 작품에 대한 전문적인 정보를 상세하고 구체적으로 알고 싶어 하는 경우에 호출합니다.
    """
    from ast import literal_eval
    llm = ChatUpstage()
    prompt = "아래는 유저가 검색을 하고 싶어하는 내용입니다. 아래 검색 내용에서 핵심 단어 1개만 추출해서 key는 '키워드', value는 핵심단어의 형태로 JSON으로만 출력해주세요. 키워드는 위키피디아에 있는 엔티티여야 합니다.\n검색 내용: {user}\n\n요약 내용:\n"
    prompt = "아래는 유저의 질문 내용입니다. 질문에서 검색을 위한 핵심 단어 1개만 답해주세요. 핵심 단어는 키워드는 위키피디아에 있는 엔티티여야 합니다. 코드를 사용하거나 다른 답변을 하지말고 바로 핵심 단어만 대답해야 합니다.\n검색 내용: {user}\n\n핵심 단어:\n"
    input_prompt = prompt.format(user=query)
    output = llm.invoke(input_prompt).content.strip()
    # try:
    #     output = literal_eval(output)
    #     query = output['키워드'].strip()
    # except:
    #     output = llm.invoke(prompt).content.strip()
    #     output = literal_eval(output)
    # query = output['키워드'].strip()
    wiki = wikipediaapi.Wikipedia(
        language='ko',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent='my-custom-user-agent')
    
    wiki_page = wiki.page(title=output)
    summary = wiki_page.summary
    if wiki_page.exists() == False:
        return "위키피디아 검색에 실패하였습니다."
    else:
        return summary[:2000]

@tool
def archiving(query: str) -> str:
    """사용자와 나눈 대화를 저장합니다. 사용자가 대화를 저장해서 SNS 등에 공유하고 싶어하는 경우에 호출합니다.
    """
    return "CALL ARCHIVING"