import streamlit as st
import openai
import base64
import mimetypes
import os
from PIL import Image
from llama_index.core import StorageContext, load_index_from_storage

# ✅ OpenAI APIキー
import os
openai.api_key = os.getenv("OPENAI_API_KEY")


# ✅ Base64画像変換（アバター用）
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

avatar_base64 = get_base64_image("のりfitnessAI (1).png")

# ✅ アップロード画像 → base64 URL
def image_to_base64_str(uploaded_file):
    mime = mimetypes.guess_type(uploaded_file.name)[0]
    uploaded_file.seek(0)
    base64_str = base64.b64encode(uploaded_file.read()).decode()
    return f"data:{mime};base64,{base64_str}"

# ✅ LlamaIndex 読み込み
@st.cache_resource
def load_query_engine():
    index_path = "./data"
    if not os.path.exists(index_path):
        st.error("❌ RAG用のindexフォルダ（./data）が見つかりません。Colabなどで作成して配置してください。")
        st.stop()
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)
    return index.as_query_engine()

query_engine = load_query_engine()

# ✅ UI 設定
st.set_page_config(page_title="のりfitnessAI", layout="centered")
st.image("のりfitnessAI.png", use_container_width=True)
st.title("のりfitnessAI")
st.markdown("📸 **食事や筋トレフォームの画像があればアップしてね！**")

# ✅ セッション初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ 画像アップロード処理
uploaded_images = st.file_uploader("画像をアップロード（複数可）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
image_data_urls = []
if uploaded_images:
    for img in uploaded_images:
        st.image(img, use_container_width=True)
        image_data_urls.append(image_to_base64_str(img))

# ✅ ユーザー入力受付
user_input = st.chat_input("質問やコメントをどうぞ！")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("考え中...少々お待ちください..."):
        try:
            rag_result = query_engine.query(user_input)
            rag_text = rag_result.response
        except Exception as e:
            rag_text = "❌ 論文ベースの回答取得でエラーが発生しました。"

        system_prompt = {
            "role": "system",
            "content": """
あなたは「のりfitness」という理論派トレーナーです。
以下のルールに従って、初心者にも分かりやすく、科学的根拠に基づいたフィットネスアドバイスを提供してください。

【ルール】

1. 専門用語は極力使わず、初心者でも理解できる言葉で説明してください。
   → やむを得ず使う場合は、必ず簡単な説明を添えてください。

2. ユーザーが画像をアップロードした場合は、内容を推測して、筋トレフォームや食事内容に対する具体的な改善アドバイスをしてください。

3. 科学的根拠（論文や研究）がある場合は、可能な限り**複数の研究**を引用してください。
   → 「2023年の研究では〜」「別の研究でも同様の結果が出ています」など、年を明記しながら説明してください。
   → 最新の研究（過去5年以内）を優先的に引用し、それ以前の研究と比較しても構いません。

4. 難しい話も、たとえ話・比喩を交えて分かりやすく！
   → 例：「筋肉は貯金と一緒です。コツコツ貯めていくことで増えていきます！」のように、親しみやすく伝えてください。

5. 全体のトーンは、明るく前向きに！
   → まるでジムで隣にいてくれる頼れる兄貴のように、元気づけながら熱く指導してください。
"""
        }

        user_content = [{"type": "text", "text": f"{user_input}\n\n以下は論文ベースの情報です：\n{rag_text}"}]
        for image_url in image_data_urls:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[system_prompt, {"role": "user", "content": user_content}],
                max_tokens=2000,
            )
            assistant_reply = response.choices[0].message.content
        except Exception as e:
            assistant_reply = f"⚠️ ChatGPT応答でエラーが発生しました: {e}"

        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# ✅ チャット履歴表示
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧍‍♂️ あなた:** {msg['content']}")
    else:
        st.markdown(f"""
        <div style='display: flex; align-items: flex-start; margin-top: 10px;'>
            <img src="data:image/png;base64,{avatar_base64}" width="40" style="border-radius: 50%; margin-right: 10px;" />
            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; max-width: 85%;'>
                {msg["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
