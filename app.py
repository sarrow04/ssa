import streamlit as st
import pandas as pd
import re
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import os

# --- モデルの読み込み（初回のみ実行・キャッシュ） ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    clf = None
    if os.path.exists('model.pkl'):
        clf = lgb.Booster(model_file='model.pkl')
    return embedder, clf

embedder, clf = load_models()

# --- 前処理関数 ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]*>', '', text)
    return text.strip()

st.title('AIプロンプトセキュリティ判定システム')

tab1, tab2 = st.tabs(['テキスト直接入力', 'CSVファイル一括判定'])

# --- タブ1: 1件ずつの判定 ---
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

# --- タブ2: テキストデータ（CSV）の読み込み判定 ---
with tab2:
    uploaded_file = st.file_uploader('判定したいテキストを含むCSVをアップロード', type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('読み込んだデータプレビュー:', df.head(3))
        
        # テキストが含まれる列名を選択
        text_col = st.selectbox('判定対象のテキスト列を選択してください', df.columns)
        
        if st.button('CSV内データを一括判定'):
            if clf is not None:
                with st.spinner('ベクトル化およびLightGBM推論中...'):
                    # 1. 前処理
                    cleaned_texts = df[text_col].apply(clean_text).tolist()
                    
                    # 2. ベクトル化
                    vectors = embedder.encode(cleaned_texts)
                    
                    # 3. LightGBM推論
                    preds = clf.predict(pd.DataFrame(vectors))
                    
                    # 結果を元のデータフレームに追加
                    df['risk_score'] = preds
                    df['is_malicious'] = (preds >= 0.5).astype(int)
                    
                    st.success('判定が完了しました！')
                    st.dataframe(df)
                    
                    # 判定結果CSVのダウンロードボタン
                    csv_data = df.to_csv(index=False).encode('utf-8_sig')
                    st.download_button(
                        label='判定結果CSVをダウンロード',
                        data=csv_data,
                        file_name='prediction_results.csv',
                        mime='text/csv'
                    )
            else:
                st.warning('model.pkl が配置されていません。')
