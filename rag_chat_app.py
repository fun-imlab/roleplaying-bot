import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.faiss import FAISS as LCFAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import base64
import os
import math



# --- Google Sheets認証セットアップ（1回だけでOK） ---
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
gc = gspread.authorize(creds)

# --- 書き込み先スプレッドシートIDを指定（URLの/d/と/editの間の部分）---
SPREADSHEET_ID = "1C3roVQgqCNQCjEsZCcI7zY1UXEp5fboQDAa9ERDufFY"
sh = gc.open_by_key(SPREADSHEET_ID)
worksheet = sh.sheet1  # 1枚目のシートを使う


# --- Streamlit UI ---
st.set_page_config(page_title="人工知能基礎 角先生Bot", layout="wide")
st.title("🎓 人工知能基礎 角先生Bot")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI APIキーが見つかりません。")
os.environ["OPENAI_API_KEY"] = openai_api_key

# --- 画像ファイル（sumi.jpeg）をBase64形式で読み込み ---
def load_image_as_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = load_image_as_base64("sumi.jpeg")  # ← 先生用の画像を指定

# --- チャンク単位でembeddingをバッチ送信し、embedding済みベクトルをFAISSに登録 ---
def batch_faiss_from_documents(docs, embedding, batch_size=400):
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    all_embeddings = []
    total = len(texts)
    batches = math.ceil(total / batch_size)
    for i in range(batches):
        start = i * batch_size
        end = min(start + batch_size, total)
        batch = texts[start:end]
        batch_embeds = embedding.embed_documents(batch)
        all_embeddings.extend(batch_embeds)

    # FAISSへの登録（最新版に対応）
    text_embeddings = list(zip(texts, all_embeddings))
    vectordb = LCFAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedding,
        metadatas=metadatas
    )
    return vectordb

# --- RAG用ベクトルDB構築 ---
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("rag_trainning.txt", encoding="utf-8")
    documents = loader.load()
    splitter = CharacterTextSplitter(
        separator="。",
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)
    embedding = OpenAIEmbeddings()
    vectordb = batch_faiss_from_documents(docs, embedding, batch_size=400)
    return vectordb

vectordb = load_vectorstore()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

template = """
あなたはA先生本人として、講義に参加した学生からの質問に答えます。
口調・語尾・話し方の癖・思考の特徴などは、以下の講義テキストから忠実に学び、再現してください。
第何回のなんというタイトルの授業での話題かも合わせて答えてください。

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

llm = ChatOpenAI(model_name="gpt-4.1") 
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# --- セッションにチャット履歴を保持 ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- 入力フォーム ---
with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input("💬 講義に基づいて質問してみてください")
    submitted = st.form_submit_button("送信")

if submitted and query:
    with st.spinner("考え中..."):
        result = qa(query)
        st.session_state.history.append({
            "query": query,
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        })

        # 現在時刻を日本時間で取得（例: 2024-07-24 20:25:33）
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Google Sheetsに「時刻」と「質問文」だけを追加
        worksheet.append_row([now_str, query])

# --- チャット履歴を新しい順に上から表示する ---
for idx, chat in reversed(list(enumerate(st.session_state.history))):
    with st.container():
        st.markdown(f"#### 🧑‍🎓 質問 {idx+1}")
        st.write(chat["query"])

        # ✅【先生のアイコン画像を指定の画像に変更】
        st.markdown(
            """
            <div style='display: flex; align-items: center;'>
                <img src='data:image/jpeg;base64,{}' width='50' height='50' style='border-radius:50%;'>
                <strong style='font-size:20px; margin-left:10px;'>角先生の回答：</strong>
            </div>
            """.format(img_base64),
            unsafe_allow_html=True
        )

        st.success(chat["answer"])

        # 参考文書をexpanderで
        with st.expander("▶️ 参考に使われた講義テキストを見る"):
            for s_idx, src in enumerate(chat["sources"]):
                st.markdown(f"**{s_idx+1} :** {src}")

        st.markdown("---")
