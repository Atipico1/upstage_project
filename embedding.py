from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from mecab import MeCab
import json
import os

class Search():
    def __init__(self, art_df, ehb_df):
        # 임베딩이 존재하지 않는 경우에만 인덱싱
        if os.path.exists("./chroma_db"):
            print("already exist db")
            vectorstore_full = Chroma(persist_directory="./chroma_db", 
                                collection_name="full_art_index",
                                embedding_function=UpstageEmbeddings(model="solar-embedding-1-large"))
    
            self.retriever_full = vectorstore_full.as_retriever(search_kwargs={'k': 6})

            vectorstore_art = Chroma(persist_directory="./chroma_db", 
                                collection_name="art_index",
                                embedding_function=UpstageEmbeddings(model="solar-embedding-1-large"))
            self.retriever_art = vectorstore_art.as_retriever(search_kwargs={'k': 4})

        else:
            print("Embedding Docs")
            # 작가와 작품설명 임베딩
            self.retriever_full = self.art_embedding("full_art_index", art_df, art_df['작품 설명'].tolist()) 
            
            # 작품 설명 임베딩
            self.retriever_art = self.art_embedding("art_index", art_df, art_df['작품 설명'].tolist())
        
        # 작품 설명 임베딩
        self.retriever_ehb = self.ehb_embedding(ehb_df)
        
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
    
    def ehb_embedding(self, df):
        docs_path = "sparse_db/ehb_docs.json"

        if not os.path.exists(docs_path):
            mecab = MeCab()
            splited_titles = df['전시'].apply(lambda x: " ".join(mecab.morphs(x))).tolist()
            indexs = df['번호'].tolist()
            titles = df['전시'].tolist()
            art_list = df['작품리스트'].tolist()
            

            docs = [
                Document(
                    page_content=splited_title,
                    metadata={"index": index, "title": title, "art_list": art_list}
                )
                for splited_title, index, title, art_list in zip(splited_titles, indexs, titles, art_list)
            ]

            retriever = BM25Retriever.from_documents(docs)

            dir_path = os.path.dirname(docs_path)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            with open(docs_path, 'w') as f:
                for doc in docs:
                    f.write(doc.json() + '\n')
        else:
            docs = []
            with open(docs_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    obj = Document(**data)
                    docs.append(obj)

            retriever = BM25Retriever.from_documents(docs)

        return retriever



    def search(self, query, retrieval):
        docs = retrieval.invoke(query)
        return docs
    
    def bm_search(self, query, retrieval):
        mecab = MeCab()
        query = " ".join(mecab.morphs(query))
        docs = retrieval.invoke(query)
        return docs
        