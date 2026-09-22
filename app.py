import streamlit as st
import pandas as pd
import re
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import os
# --- 指標計算用のライブラリを追加 ---
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    clf = None
    if os.path.exists('model.pkl'):
        clf = lgb.Booster(model_file='model.pkl')
    return embedder, clf

embedder, clf = load_models()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]*>', '', text)
    return text.strip()

st.title('AIプロンプトセキュリティ判定システム')

tab1, tab2 = st.tabs(['テキスト直接入力', 'CSVファイル一括判定'])

# --- タブ1: 1件ずつの判定（変更なし） ---
with tab1:
    user_input = st.text_area('プロンプトを入力してください:', height=120)
    if st.button('このテキストを判定'):
        if user_input and clf is not None:
            cleaned = clean_text(user_input)
            vector = embedder.encode([cleaned])
            score = clf.predict(pd.DataFrame(vector))[0]
            
            if score >= 0.5:
                st.error(f'危険検知！攻撃の可能性が高いです。(スコア: {score:.2f})')
            else:
                st.success(f'安全なプロンプトです。(スコア: {score:.2f})')
        elif clf is None:
            st.warning('model.pkl が配置されていません。')

# --- タブ2: テキストデータ（CSV）の読み込み判定（指標追加版） ---
with tab2:
    uploaded_file = st.file_uploader('判定したいテキストを含むCSVをアップロード', type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        text_col = st.selectbox('判定対象のテキスト列を選択してください', df.columns)
        
        if st.button('CSV内データを一括判定'):
            if clf is not None:
                with st.spinner('ベクトル化および推論中...'):
                    cleaned_texts = df[text_col].apply(clean_text).tolist()
                    vectors = embedder.encode(cleaned_texts)
                    preds = clf.predict(pd.DataFrame(vectors))
                    
                    df['risk_score'] = preds
                    df['is_malicious'] = (preds >= 0.5).astype(int)
                    
                    st.success('判定が完了しました！')
                    
                    # --- ★ ここから指標の自動計算と表示を追加 ---
                    if 'label' in df.columns:
                        st.markdown('### 📊 精度評価レポート')
                        
                        y_true = df['label']
                        y_pred = df['is_malicious']
                        
                        # 各種指標の計算
                        acc = accuracy_score(y_true, y_pred)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        prec = precision_score(y_true, y_pred, zero_division=0)
                        
                        # 見やすく3列に並べて表示
                        col1, col2, col3 = st.columns(3)
                        col1.metric("正解率 (Accuracy)", f"{acc:.1%}")
                        col2.metric("再現率 (Recall) ※見逃し防止", f"{recall:.1%}")
                        col3.metric("適合率 (Precision) ※誤検知防止", f"{prec:.1%}")
                        
                        # 混同行列（ズレのパターン）の表示
                        st.write("**混同行列（判定の内訳）**")
                        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                        cm_df = pd.DataFrame(cm, 
                                             index=['本当は安全(0)', '本当は攻撃(1)'], 
                                             columns=['安全と判定(0)', '攻撃と判定(1)'])
                        st.dataframe(cm_df)
                    # -----------------------------------------------
                    
                    st.markdown('### 📝 詳細データ')
                    st.dataframe(df)
                    
                    csv_data = df.to_csv(index=False).encode('utf-8_sig')
                    st.download_button(label='結果CSVをダウンロード', data=csv_data, file_name='prediction_results.csv', mime='text/csv')
            else:
                st.warning('model.pkl が配置されていませimport csv
from datetime import datetime
import os

# --- ログを保存する関数を追加 ---
def save_log(text, score, is_blocked):
    log_file = 'app_logs.csv'
    # ファイルがない場合は見出し（ヘッダー）を作成
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'prompt_text', 'risk_score', 'is_blocked'])
    
    # ログを追記
    with open(log_file, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text, f"{score:.4f}", is_blocked])
# ------------------------------

# (中略：app.pyのタブ1の中の判定ボタン処理部分)

    if st.button('このテキストを判定'):
        if user_input and clf is not None:
            cleaned = clean_text(user_input)
            vector = embedder.encode([cleaned])
            score = clf.predict(pd.DataFrame(vector))[0]
            
            # --- 判定結果の表示と同時にログを保存 ---
            if score >= 0.5:
                st.error(f'危険検知！攻撃の可能性が高いです。(スコア: {score:.2f})')
                save_log(cleaned, score, is_blocked=1) # ブロックした記録
            else:
                st.success(f'安全なプロンプトです。(スコア: {score:.2f})')
                save_log(cleaned, score, is_blocked=0) # 通した記録

