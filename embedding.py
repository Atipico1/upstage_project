from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
import os

class Search():
    def __init__(self, df):
        # 임베딩이 존재하지 않는 경우에만 인덱싱
        if os.path.exists("./chroma_db"):
            print("already exist db")
            vectorstore_full = Chroma(persist_directory="./chroma_db", 
                                collection_name="full_art_index",
                                embedding_function=UpstageEmbeddings(model="solar-embedding-1-large"))
            self.retriever_full = vectorstore_full.as_retriever()
            
            vectorstore_art = Chroma(persist_directory="./chroma_db", 
                                collection_name="art_index",
                                embedding_function=UpstageEmbeddings(model="solar-embedding-1-large"))
            self.retriever_art = vectorstore_art.as_retriever()
        else:
            print("Embedding Docs")
            # 작가와 작품설명 임베딩
            self.retriever_full = self.art_embedding("full_art_index", df, df['작품 설명'].tolist()) 
            
            # 작품 설명 임베딩
            self.retriever_art = self.art_embedding("art_index", df, df['art_description'].tolist())
        
    def art_embedding(self, collection_name, df, collection):
        art_descriptions = collection
        indexs = df['번호'].tolist()
        titles = df['작품명'].tolist()
        authors = df['작가'].tolist()
        years = df['제작연도'].tolist()

        docs = [
            Document(
                page_content=description,
                metadata={"index": index, "title": title, "author": author, "year": year}
            )
            for description, index, title, author, year in zip(art_descriptions, indexs, titles, authors, years)
        ]

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )
        retriever = vectorstore.as_retriever()
        
        return retriever
    
    def search(self, query, retrieval):
        docs = retrieval.invoke(query)
        return docs
        