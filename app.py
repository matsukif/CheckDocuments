import os
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter

uploaded_file = st.sidebar.file_uploader("upload", type="pdf")
os.environ['OPENAI_API_KEY'] = st.secrets.OpenAIAPI.openai_api_key #StreamlitのSecretsからAPI keyをとってくる
#openai_api_key = st.secrets.OpenAIAPI.openai_api_key #StreamlitのSecretsからAPI keyをとってくる

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size = 2000,
  chunk_overlap = 100,
  length_function = len,
)

if uploaded_file :
  with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    tmp_file.write(uploaded_file.getvalue())
    tmp_file_path = tmp_file.name

  loader = PyPDFLoader(file_path=tmp_file_path)  
  data = loader.load_and_split(text_splitter)

  #ドキュメントから関連情報を高速に検索するためにFAISS Vectorstoreを利用
  embeddings = OpenAIEmbeddings()
  vectors = FAISS.from_documents(data, embeddings)

  #過去の質問や回答内容を加味してテキスト出力するためにConversationalRetrievalChainを利用
  chain = ConversationalRetrievalChain.from_llm(llm= ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo-0613'),
                                               retriever=vectors.as_retriever()
                                              ) 

  #会話履歴をConversationalRetrievalChainに渡すことで過去の応答や読み込んだファイルから回答を生成する
  def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

  #UX向上のため「セッション初期化」「メッセージ表示」を行う
  if 'history' not in st.session_state: #セッションを初期化するためにst.session_state['history']の通りに宣言
    st.session_state['history'] = []

  if 'generated' not in st.session_state: #['generated']にてモデルから生成された回答を保存
    st.session_state['generated'] = ["このドキュメントについてなんでも聞いてね\n" + uploaded_file.name]
  
  if 'past' not in st.session_state: #['past']はユーザーが入力したメッセージを保存
    st.session_state['past'] = ["Hey !"]
  
  response_container = st.container() #コンテナは必須ではないが、チャットメッセージの下に質問エリアを配置することでＵＸ向上
  container = st.container()

  #session.stateとコンテナを設定すればユーザーが質問を入力して回答できる状態になる。
  #button押下時、conversatinal_chat関数が呼ばれ['generated']と['past']にたいして入力した質問と生成された回答が保存される
  with container:
    with st.form(key='my_form', clear_on_submit=True):

      user_input = st.text_input("input:", placeholder="Talk about your pdf data.", key='imput')
      submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input :
      output =  conversational_chat(user_input) 

      st.session_state['past'].append(user_input)
      st.session_state['generated'].append(output) 

  #stream_chatモジュールでユーザーとチャットボット間のやり取りをStremlitサイト上に表示
  if st.session_state['generated']:
    with response_container:
      for i in range(len(st.session_state['generated'])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
        message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
