# =========================
# のりfitnessAI（RAG一貫化 + 会話メモリ + 引用可視化 + 診断UI + 年明示）
# =========================
import os
import json
import re
import base64
import mimetypes
from pathlib import Path
import streamlit as st
from PIL import Image

# ===== OpenAI =====
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("openai", {}).get("api_key")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY が設定されていません。`.streamlit/secrets.toml` または環境変数で設定してください。")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== LlamaIndex =====
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.response_synthesizers import get_response_synthesizer

# ========= ユーティリティ =========
def get_base64_image(image_path: str) -> str:
    if not os.path.exists(image_path): return ""
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

def image_to_base64_str(uploaded_file) -> str:
    mime = mimetypes.guess_type(uploaded_file.name)[0] or "image/png"
    uploaded_file.seek(0)
    b64 = base64.b64encode(uploaded_file.read()).decode()
    return f"data:{mime};base64,{b64}"

def build_openai_messages(history, latest_user_content):
    msgs = []
    for m in history[-16:]:
        role, content = m["role"], m["content"]
        if isinstance(content, list):
            text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            content = "\n".join(text_parts) if text_parts else ""
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": latest_user_content})
    return msgs

# ---- 年を抽出するヘルパー ----
YEAR_RE = re.compile(r"(20\d{2})年?|(?:\()?(20\d{2})(?:\))?")

def _pick_year_from_text(text: str) -> str | None:
    if not text: return None
    m = YEAR_RE.search(text)
    if m:
        return m.group(1) or m.group(2)
    return None

def extract_year(meta: dict, preview_text: str) -> str | None:
    for k in ("file_name", "document_id", "doc_id", "source", "title"):
        if k in meta and isinstance(meta[k], str):
            y = _pick_year_from_text(meta[k])
            if y: return y
    y = _pick_year_from_text(preview_text)
    return y

def fmt_sources(nodes, max_items=3, with_preview=True):
    lines = []
    for n in nodes[:max_items]:
        meta = n.node.metadata or {}
        preview = (n.node.get_content() or "").replace("\n", " ")
        year = extract_year(meta, preview) or "----"
        file_or_id = meta.get("file_name") or meta.get("document_id") or "unknown"
        page = meta.get("page_label") or meta.get("page") or ""
        score = getattr(n, "score", None)
        score_txt = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
        page_txt = f" p.{page}" if page else ""
        if with_preview:
            lines.append(f"- [{year}] score={score_txt} {file_or_id}{page_txt}\n  └ preview: {preview[:180]}…")
        else:
            lines.append(f"- [{year}] score={score_txt} {file_or_id}{page_txt}")
    return "\n".join(lines)

# ---- 埋め込み次元を自動判定 ----
def detect_index_dim(persist_dir: str) -> int | None:
    vf = Path(persist_dir) / "default__vector_store.json"
    if not vf.exists(): return None
    try:
        with open(vf, "r", encoding="utf-8") as f:
            data = json.load(f)
        emb_dict = data.get("embedding_dict") or data.get("embeddings") or {}
        if isinstance(emb_dict, dict) and emb_dict:
            first_vec = next(iter(emb_dict.values()))
            if isinstance(first_vec, list):
                return len(first_vec)
        dim = (data.get("metadata") or {}).get("embedding_dim")
        return int(dim) if isinstance(dim, int) else None
    except Exception:
        return None

# ========= RAG ローダ =========
INDEX_DIR = "./data"

@st.cache_resource
def load_rag(_rev: str):
    if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
        raise FileNotFoundError(f"❌ index が見つかりません: {os.path.abspath(INDEX_DIR)}")

    detected_dim = detect_index_dim(INDEX_DIR)
    if detected_dim == 3072:
        embed_model_name = "text-embedding-3-large"
    elif detected_dim == 1536:
        embed_model_name = "text-embedding-ada-002"  # 旧互換
    else:
        embed_model_name = "text-embedding-ada-002"

    Settings.embed_model = OpenAIEmbedding(model=embed_model_name, api_key=OPENAI_API_KEY)

    # ✅ 同期ロードを維持
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)

    # ✅ recall を強化
    retriever = index.as_retriever(similarity_top_k=30)
    post = SimilarityPostprocessor(similarity_cutoff=0.40)  # 少し緩める
    synth = get_response_synthesizer(response_mode="tree_summarize")

    try:
        node_count = len(index.docstore.docs)  # type: ignore[attr-defined]
    except Exception:
        node_count = -1

    health = {
        "persist_dir": os.path.abspath(INDEX_DIR),
        "node_count": node_count,
        "detected_dim": detected_dim,
        "embed_model": embed_model_name,
    }
    return retriever, post, synth, health

# ========= UI 初期化 =========
st.set_page_config(page_title="のりfitnessAI", layout="centered")
avatar_base64 = get_base64_image("のりfitnessAI (1).png")
st.image("のりfitnessAI.png", use_container_width=True)
st.title("のりフィットネスAI")
st.markdown("📸 **食事や筋トレフォームの画像があればアップしてね！**")

# セッション初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========= RAG 読み込み =========
try:
    retriever, postproc, synthesizer, rag_health = load_rag(os.getenv("K_REVISION", "local"))
except Exception as e:
    st.error(f"❌ RAGの初期化に失敗しました: {e}")
    st.stop()

# ========= 開発モード判定 =========
env_from_secrets = (st.secrets.get("app", {}) or {}).get("env")
debug_qp = None
try:
    debug_qp = st.query_params.get("debug")
except Exception:
    pass

DEV_MODE = (
    (os.getenv("APP_ENV") or env_from_secrets or "production") != "production"
) or (str(debug_qp).lower() in ("1", "true"))

strict = False

# ========= サイドバー =========
if DEV_MODE:
    with st.sidebar:
        st.header("🔧 Debug / RAG Health")
        st.write("K_REVISION:", os.getenv("K_REVISION", "local"))
        st.write("INDEX_DIR:", rag_health.get("persist_dir"))
        st.write("node_count:", rag_health.get("node_count"))
        st.write("detected_dim:", rag_health.get("detected_dim"))
        st.write("embed_model:", rag_health.get("embed_model"))
        masked = OPENAI_API_KEY[:4] + "…" + OPENAI_API_KEY[-4:]
        st.write("OPENAI_API_KEY:", masked)

        strict = st.toggle("RAG厳格モード（根拠なしなら回答もしない）", value=False)

        st.subheader("🔎 診断クエリ")
        diag_q = st.text_input("例: タンパク質 高齢 1.5倍", value="人工甘味料 体重")
        tmp_cutoff = st.slider("similarity_cutoff（診断用）", 0.0, 0.9, 0.40, 0.05)
        if st.button("実行"):
            try:
                raw_hits = retriever.retrieve(diag_q)
                st.write("raw_hits:", len(raw_hits))
                diag_post = SimilarityPostprocessor(similarity_cutoff=tmp_cutoff)
                filtered = diag_post.postprocess_nodes(raw_hits)
                st.write("filtered_hits:", len(filtered))
                st.code(fmt_sources(filtered if filtered else raw_hits, max_items=5), language="text")
            except Exception as e:
                st.write("診断エラー:", e)

        if st.button("🧹 RAGキャッシュをクリア"):
            load_rag.clear()
            st.success("キャッシュをクリアしました。ページを再読み込みしてください。")

# ========= 画像アップロード =========
uploaded_images = st.file_uploader("画像をアップロード（複数可）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
image_data_urls = []
if uploaded_images:
    for img in uploaded_images:
        st.image(img, use_container_width=True)
        image_data_urls.append(image_to_base64_str(img))

# ========= チャット入力 =========
user_input = st.chat_input("質問やコメントをどうぞ！")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("考え中...少々お待ちください..."):
        try:
            raw_nodes = retriever.retrieve(user_input)
            nodes = postproc.postprocess_nodes(raw_nodes)

            years = []
            for n in nodes[:5]:
                meta = n.node.metadata or {}
                preview = (n.node.get_content() or "")
                y = extract_year(meta, preview)
                if y: years.append(y)
            years_uniq = sorted(set(years), reverse=True)

            if nodes:
                rag_result = synthesizer.synthesize(query=user_input, nodes=nodes)
                rag_text = getattr(rag_result, "response", None) or str(rag_result) or ""
                sources_block = fmt_sources(nodes, max_items=3, with_preview=True)
                if years_uniq:
                    rag_text += "\n\n**年ヒント:** " + ", ".join([f"{y}年" for y in years_uniq])
                rag_text += "\n\n---\n**参照元**\n" + sources_block
            else:
                rag_text = "（関連する根拠ドキュメントが見つかりませんでした）"

        except Exception as e:
            rag_text = f"❌ 論文ベースの回答取得でエラー：{e}"
            nodes = []

        system_prompt = (
            "あなたは『のりfitness』という理論派トレーナーです。"
            "以下の『コンテキスト』に基づいて回答してください。"
            "必ず少なくとも一度は『YYYY年の研究では〜』という形で年を明示し、"
            "複数年（例: 2025年・2023年）があれば自然に盛り込むこと。"
            "根拠が無い場合は『ここからは根拠なしですが』と明示し、推測で断定しないこと。"
            "トーンは明るく、頼れる兄貴のように。"
        )

        if strict and not nodes:
            assistant_reply = "根拠ドキュメントが無いため回答を控えます（RAG厳格モード）。"
        else:
            latest_user_content = [
                {"type": "text",
                 "text": (
                     f"質問: {user_input}\n\n---\n"
                     f"コンテキスト:\n{rag_text}\n"
                     f"---\nこの範囲で回答してください。"
                 )}]
            for url in image_data_urls:
                latest_user_content.append({"type": "image_url", "image_url": {"url": url}})

            try:
                messages = [{"role": "system", "content": system_prompt}]
                messages += build_openai_messages(st.session_state.messages[:-1], latest_user_content)
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.2,
                    max_tokens=1800,
                    messages=messages,
                )
                assistant_reply = resp.choices[0].message.content
            except Exception as e:
                assistant_reply = f"⚠️ ChatGPT応答でエラー: {e}"

        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# ========= チャット履歴描画 =========
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧍‍♂️ あなた:** {msg['content']}")
    else:
        st.markdown(
            f"""
            <div style='display: flex; align-items: flex-start; margin-top: 10px;'>
                <img src="data:image/png;base64,{avatar_base64}" width="40"
                     style="border-radius: 50%; margin-right: 10px;" />
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; max-width: 85%;'>
                    {msg["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
