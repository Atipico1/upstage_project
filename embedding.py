import pandas as pd
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain.docstore.document import Document

def prepare_embed():
    df = pd.read_csv("data/arts02.csv")

    art_descriptions = df['작품 설명'].tolist()
    indexs = df['번호'].tolist()
    titles = df['작품명'].tolist()
    authors = df['작가'].tolist()
    years = df['제작연도'].tolist()

    art_full_docs = [
        Document(
            page_content=description,
            metadata={"indexs": index, "title": title, "author": author, "year": year}
        )
        for description, index, title, author, year in zip(art_descriptions, indexs, titles, authors, years)
    ]


    vectorstore = Chroma.from_documents(
        documents=art_full_docs,
        embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
        collection_name="full_art_index",
    )
    
    retriever_full = vectorstore.as_retriever()
    return retriever_full