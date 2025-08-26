import streamlit as st
import openai
import base64
from PIL import Image
import mimetypes

# APIクライアント設定
client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

# アバター画像をbase64化
def get_base64_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

avatar_base64 = get_base64_image("のりfitnessAI (1).png")

# 画像をbase64文字列に変換
def image_to_base64_str(uploaded_file):
    mime = mimetypes.guess_type(uploaded_file.name)[0]
    base64_str = base64.b64encode(uploaded_file.read()).decode()
    return f"data:{mime};base64,{base64_str}"

# ページ設定
st.set_page_config(page_title="のりfitnessAI", layout="centered")
st.image("のりfitnessAI.png", use_container_width=True)
st.title("のりfitnessAI")
st.markdown("📸 **食事・筋トレの画像があればアップしてね！**")

# セッション初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 画像アップロード（複数対応）
uploaded_images = st.file_uploader("画像をアップロードしてください（複数可）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
image_data_urls = []
if uploaded_images:
    for img in uploaded_images:
        st.image(img, use_container_width=True)
        image_data_urls.append(image_to_base64_str(img))

# チャット入力
user_input = st.chat_input("質問やコメントをどうぞ！")

if user_input:
    # ユーザー入力を保存
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("考え中...少々お待ちください！"):
        # システムプロンプト
        system_prompt = {
            "role": "system",
            "content": """
あなたは「のりfitness」という情熱的なパーソナルトレーナーであり、科学的根拠に基づいたアドバイスができる信頼される存在です。

回答は以下のように構成してください：
・初心者にも分かりやすく、専門用語を避けた説明
・高い熱量とやる気を引き出す言葉
・親しみやすい語り口
・共感・行動提案を自然に織り交ぜること
・アップロード画像の内容も積極的に読み取って評価する（筋トレフォームや食事内容など）

例：「このおにぎりはいいね！糖質はトレーニング前に最適。タンパク質ももう少し入るとベスト！」など。
"""
        }

        # ユーザーメッセージ構築
        user_content = [{"type": "text", "text": user_input}]
        for image_url in image_data_urls:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        # ChatGPTへ送信
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[system_prompt, {"role": "user", "content": user_content}],
            max_tokens=1500,
        )
        assistant_reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# チャット履歴表示
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
