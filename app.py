import streamlit as st
import pandas as pd
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import csv
from datetime import datetime
import os
import re

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
    # ファイルがない場合は新規作成してヘッダーを書き込む
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'prompt_text', 'risk_score', 'is_blocked'])
    
    # ログを追記する
    with open(log_file, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text, f"{score:.4f}", is_blocked])

# === 3. テキスト前処理関数 ===
def clean_text(text):
    # 余分な空白や改行を綺麗にする
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === 4. Streamlit UI 画面構成 ===
st.title('🛡️ AIセキュリティガードレール')

# モデルがない場合のエラーハンドリング
if clf is None:
    st.warning('⚠️ 「model.pkl」が配置されていません。Colabでダウンロードしたファイルを同じフォルダに配置してください。')
    st.stop() # ここで処理を止める

tab1, tab2 = st.tabs(["💬 テキスト直接入力", "📁 CSV一括判定"])

# --- タブ1: テキスト直接入力とログ保存 ---
with tab1:
    st.markdown("### プロンプトインジェクション判定")
    user_input = st.text_area("判定したいテキストを入力してください", height=150)
    
    if st.button('このテキストを判定'):
        if user_input:
            cleaned_text = clean_text(user_input)
            
            # ベクトル化してLightGBMでスコア算出
            vector = embedder.encode([cleaned_text])
            score = clf.predict(pd.DataFrame(vector))[0]
            
            # スコアによるブロック判定と画面表示、ログ保存
            if score >= 0.5:
                st.error(f'🚨 危険検知！攻撃の可能性が高いです。(危険スコア: {score:.4f})')
                save_log(cleaned_text, score, is_blocked=1)
            else:
                st.success(f'✅ 安全なプロンプトです。(危険スコア: {score:.4f})')
                save_log(cleaned_text, score, is_blocked=0)
        else:
            st.warning("テキストを入力してください。")

# --- タブ2: CSV一括判定（テスト・検証用） ---
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
                
                # プログレスバーの表示
                progress_bar = st.progress(0)
                
                st.text("テキストを数値化しています...")
                vectors = embedder.encode(df['cleaned_text'].tolist(), show_progress_bar=False)
                
                st.text("AIが危険度を判定中...")
                scores = clf.predict(pd.DataFrame(vectors))
                
                # 結果をDataFrameに反映
                df['risk_score'] = scores
                df['is_malicious'] = (df['risk_score'] >= 0.5).astype(int)
                
                progress_bar.progress(100)
                
                st.success("判定完了！")
                # 結果のプレビュー（先頭20件）
                st.dataframe(df[['text', 'risk_score', 'is_malicious']].head(20))
                
                # ダウンロードボタン
                csv_data = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="判定結果CSVをダウンロード",
                    data=csv_data,
                    file_name='prediction_results.csv',
                    mime='text/csv'
                )
