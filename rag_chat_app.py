import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# --- Streamlit UI ---
st.set_page_config(page_title="角先生なりきりChatBot", layout="wide")
st.title("🎓 角先生なりきりChatBot")


load_dotenv()  # .env を読み込む
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI APIキーが見つかりません。")

os.environ["OPENAI_API_KEY"] = openai_api_key


# --- RAG用ベクトルDBの構築 ---
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("rag_trainning.txt", encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embedding = OpenAIEmbeddings()
    
    # ✅ 修正箇所：persist_directory を削除（＝インメモリ動作に）
    vectordb = FAISS.from_documents(docs, embedding=embedding)
        
    return vectordb


vectordb = load_vectorstore()
retriever = vectordb.as_retriever()


template = """
あなたはA先生本人として、講義に参加した学生からの質問に答えます。
口調・語尾・話し方の癖・思考の特徴などは、以下の講義テキストから忠実に学び、再現してください。

以下はA先生が実際に講義中に話した内容です：

=========
{context}
=========

質問：
{question}

A先生として、まるで“今この場であなたが学生に語っているかのように”回答してください。
文章は自然な話し言葉で、句読点や語尾なども実際の口調に近づけてください。
"""

prompt_template = PromptTemplate.from_template(template)

# --- LLM + 検索チェーン ---
llm = ChatOpenAI(model_name="gpt-4")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# --- チャット入力UI ---
query = st.text_input("💬 講義に基づいて質問してみてください")
if query:
    with st.spinner("考え中..."):
        result = qa(query)
        st.success("✅ 回答")
        st.write(result["result"])

        # 参考文書の表示
        st.markdown("### 🔍 参考に使われた講義テキスト")
        for doc in result["source_documents"]:
            st.write(doc.page_content)
