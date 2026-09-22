import streamlit as st
import pandas as pd
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import csv
from datetime import datetime
import os
import re
# ↓ 成果率（メトリクス）を計算するためのライブラリを追加
from sklearn.metrics import accuracy_score, recall_score, precision_score

# === 1. モデルの読み込み（キャッシュして高速化） ===
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    clf = None
    if os.path.exists('model.pkl'):
        clf = lgb.Booster(model_file='model.pkl')
    return embedder, clf

embedder, clf = load_models()

# === 2. ログ保存用関数 ===
def save_log(text, score, is_blocked):
    log_file = 'app_logs.csv'
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'prompt_text', 'risk_score', 'is_blocked'])
    
    with open(log_file, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text, f"{score:.4f}", is_blocked])

# === 3. テキスト前処理関数 ===
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === 4. Streamlit UI 画面構成 ===
st.title('🛡️ AIセキュリティガードレール')

if clf is None:
    st.warning('⚠️ 「model.pkl」が配置されていません。同じフォルダに配置してください。')
    st.stop()

tab1, tab2 = st.tabs(["💬 テキスト直接入力", "📁 CSV一括判定"])

# --- タブ1: テキスト直接入力とログ保存 ---
with tab1:
    st.markdown("### プロンプトインジェクション判定")
    user_input = st.text_area("判定したいテキストを入力してください", height=150)
    
    if st.button('このテキストを判定'):
        if user_input:
            cleaned_text = clean_text(user_input)
            
            vector = embedder.encode([cleaned_text])
            score = clf.predict(pd.DataFrame(vector))[0]
            
            if score >= 0.5:
                st.error(f'🚨 危険検知！攻撃の可能性が高いです。(危険スコア: {score:.4f})')
                save_log(cleaned_text, score, is_blocked=1)
            else:
                st.success(f'✅ 安全なプロンプトです。(危険スコア: {score:.4f})')
                save_log(cleaned_text, score, is_blocked=0)
        else:
            st.warning("テキストを入力してください。")

# --- タブ2: CSV一括判定（成果率の表示付き） ---
with tab2:
    st.markdown("### 複数データの一括テスト")
    uploaded_file = st.file_uploader("テスト用CSVをアップロード（'text'列が必要です）", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if 'text' not in df.columns:
            st.error("CSVファイルに「text」という列が見つかりません。")
        else:
            st.write(f"読み込み完了: {len(df)} 件のデータを判定します...")
            
            if st.button('一括判定を開始'):
                df['cleaned_text'] = df['text'].apply(clean_text)
                progress_bar = st.progress(0)
                
                st.text("テキストを数値化しています...")
                vectors = embedder.encode(df['cleaned_text'].tolist(), show_progress_bar=False)
                
                st.text("AIが危険度を判定中...")
                scores = clf.predict(pd.DataFrame(vectors))
                
                df['risk_score'] = scores
                df['is_malicious'] = (df['risk_score'] >= 0.5).astype(int)
                progress_bar.progress(100)
                st.success("判定完了！")
                
                # === 復活した部分：正解データ（label）がある場合のみ成果率を表示 ===
                if 'label' in df.columns:
                    st.markdown("### 📊 モデルの判定精度")
                    y_true = df['label']
                    y_pred = df['is_malicious']
                    
                    acc = accuracy_score(y_true, y_pred)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    
                    # 画面に3つの数字を並べて綺麗に表示
                    col1, col2, col3 = st.columns(3)
                    col1.metric("正解率 (Accuracy)", f"{acc * 100:.1f}%")
                    col2.metric("再現率 (Recall) ※見逃し防止", f"{recall * 100:.1f}%")
                    col3.metric("適合率 (Precision) ※誤検知防止", f"{precision * 100:.1f}%")
                    st.divider()
                # =========================================================
                
                st.dataframe(df[['text', 'risk_score', 'is_malicious']].head(20))
                
                csv_data = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="判定結果CSVをダウンロード",
                    data=csv_data,
                    file_name='prediction_results.csv',
                    mime='text/csv'
                )
