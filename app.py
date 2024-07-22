import streamlit as st
import asyncio
import aiohttp
import time
import random
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objs as go
from textblob import TextBlob
import json
from datetime import datetime
import altair as alt
import re
from collections import Counter
import nltk
nltk.download('punkt')

# 環境変数を読み込む
load_dotenv()

# 定数の定義
API_ENDPOINTS = {
    "CLAUDE": "https://api.anthropic.com/v1/messages",
}
RETRY_ATTEMPTS = 5
MAX_CONCURRENT_REQUESTS = 3
BASE_DELAY = 2
MAX_RETRIES = 15
MAX_BACKOFF_DELAY = 60

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")

# キーワードリスト（カテゴリ別）
KEYWORDS = {
    "政治的思想・発言": ["政治", "発言", "批判", "論争", "抗議", "デモ", "選挙", "党", "イデオロギー"],
    "問題・事件・事故・脱税": ["逮捕", "事件", "事故", "脱税", "違法", "犯罪", "摘発", "調査", "疑惑"],
    "薬物・裏社会・暴力": ["薬物", "麻薬", "暴力", "暴行", "違法", "逮捕", "組織", "犯罪", "事件"],
    "性蔑視・ジェンダー・LGBT": ["差別", "蔑視", "セクハラ", "ハラスメント", "批判", "抗議", "論争", "LGBT"],
    "熱愛・不倫": ["不倫", "浮気", "スキャンダル", "離婚", "破局", "熱愛", "交際", "恋愛"],
    "炎上": ["炎上", "批判", "謝罪", "批判", "批判殺到", "非難", "バッシング", "問題発言"],
    "宗教": ["宗教", "カルト", "信者", "布教", "論争", "批判", "問題"],
    "ヌード": ["ヌード", "写真集", "過激", "セクシー", "露出", "批判", "問題"]
}

# セッション状態の初期化
if 'processing' not in st.session_state:
    st.session_state.processing = False

# ユーティリティ関数
def calculate_risk_score(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # センチメントを -1 (最もネガティブ) から 1 (最もポジティブ) のスケールで取得
    # これを 0 から 100 のスケールに変換し、ネガティブな方がリスクが高いとする
    sentiment_score = (1 - sentiment) * 50
    
    # 主観性も考慮する。主観性が高いほどリスクが高いと仮定
    subjectivity_score = subjectivity * 50
    
    # センチメントと主観性を組み合わせてリスクスコアを算出
    risk_score = (sentiment_score + subjectivity_score) / 2
    
    # デバッグ情報
    print(f"Text: {text[:100]}...")  # テキストの最初の100文字を表示
    print(f"Sentiment: {sentiment}, Subjectivity: {subjectivity}")
    print(f"Calculated Risk Score: {risk_score}")
    
    return risk_score

async def get_webpage_content(session, url):
    try:
        ua = UserAgent()
        headers = {'User-Agent': ua.random}
        async with session.get(url, headers=headers, timeout=10) as response:
            response.raise_for_status()
            html = await response.text()
            return strip_html_tags(html)
    except aiohttp.ClientError as e:
        st.error(f"Error fetching webpage: {e}")
        return ""

def strip_html_tags(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

async def call_claude_api_with_advanced_backoff(session, url, payload, headers):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def api_call():
        async with semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status in [429, 529]:  # レート制限エラーまたは過負荷エラー
                            delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF_DELAY)
                            st.warning(f"API overloaded or rate limited. Retrying in {delay:.2f} seconds...")
                            await asyncio.sleep(delay)
                            continue
                        response.raise_for_status()
                        return await response.json()
                except aiohttp.ClientResponseError as e:
                    if e.status == 529:
                        delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF_DELAY)
                        st.warning(f"API overloaded. Retrying in {delay:.2f} seconds...")
                        await asyncio.sleep(delay)
                    elif attempt == MAX_RETRIES - 1:
                        raise
                    else:
                        delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF_DELAY)
                        st.warning(f"API call failed. Retrying in {delay:.2f} seconds... Error: {e}")
                        await asyncio.sleep(delay)
                except aiohttp.ClientError as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF_DELAY)
                    st.warning(f"API call failed. Retrying in {delay:.2f} seconds... Error: {e}")
                    await asyncio.sleep(delay)
        raise Exception("Max retries reached")

    return await api_call()

async def collect_risk_info(session, talent_name, risk_word, max_results, api_key):
    url = "https://api.tavily.com/search"
    query = f"{talent_name} {risk_word}"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "include_images": False,
        "include_raw_content": False,
        "max_results": max_results,
        "include_domains": [],
        "exclude_domains": []
    }
    try:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientError as e:
        st.error(f"Error with Tavily API: {e}")
        return None
    
def calculate_advanced_risk_score(text, risk_category):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    word_counts = Counter(word for word in text.lower().split() if word in KEYWORDS[risk_category])
    keyword_score = sum(word_counts.values()) * 10

    numbers = re.findall(r'\d+', text)
    number_score = sum(int(num) for num in numbers if int(num) > 10) * 0.5

    context_phrases = [
        "謝罪", "批判", "問題", "炎上", "逮捕", "疑惑", "スキャンダル", 
        "降板", "契約解除", "引退", "活動休止", "謹慎"
    ]
    context_score = sum(10 for phrase in context_phrases if phrase in text)

    complexity_score = len(text.split()) * 0.1
    sentiment_score = (1 - sentiment) * 25
    subjectivity_score = subjectivity * 25

    total_score = (
        keyword_score + 
        number_score + 
        context_score + 
        complexity_score + 
        sentiment_score + 
        subjectivity_score
    )

    normalized_score = min(100, max(0, total_score))

    return normalized_score, {
        "keyword_score": keyword_score,
        "number_score": number_score,
        "context_score": context_score,
        "complexity_score": complexity_score,
        "sentiment_score": sentiment_score,
        "subjectivity_score": subjectivity_score
    }


async def summarize_risk_content(session, content, risk_word, max_length, model, api_key):
    url = API_ENDPOINTS["CLAUDE"]
    prompt = f"以下の文章から{risk_word}に関連する内容を{max_length}文字程度で要約してください。特にネガティブな内容や潜在的なリスクに注目してください。関連する内容がない場合は「関連情報なし」と記載してください。\n\n{content}"
    payload = {
        "model": model,
        "max_tokens": min(max_length, 4096),
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    try:
        response_data = await call_claude_api_with_advanced_backoff(session, url, payload, headers)
        if 'content' in response_data:
            return response_data['content'][0]['text']
        else:
            st.error(f"Unexpected response format: {response_data}")
            return ""
    except Exception as e:
        st.error(f"Failed to summarize content: {e}")
        return ""

async def generate_risk_report(session, talent_name, risk_summaries, model, api_key, articles):
    url = API_ENDPOINTS["CLAUDE"]
    summaries_text = "\n\n".join([f"{word}:\n{summary}" for word, summary in risk_summaries.items()])
    articles_info = "\n".join([f"[{a['index']}] {a['title']} - {a['url']} (Risk: {a['risk_word']})" for a in articles])
    prompt = f"""以下は{talent_name}に関する各リスクワードの要約です。これらの情報を基に、{talent_name}の炎上リスクを評価し、300文字程度のレポートを作成してください。
レポートには全体的なリスク評価と、特に注意すべき点を含めてください。また、重要な情報の出典として、適切な記事番号（[index]の形式）を括弧書きで示してください。

要約:
{summaries_text}

記事一覧:
{articles_info}
"""
    
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    try:
        response_data = await call_claude_api_with_advanced_backoff(session, url, payload, headers)
        if 'content' in response_data:
            return response_data['content'][0]['text']
        else:
            st.error(f"Unexpected response format: {response_data}")
            return ""
    except Exception as e:
        st.error(f"Failed to generate risk report: {e}")
        return ""

async def process_talent_risks(session, TAVILY_API_KEY, CLAUDE_API_KEY, talent_name, max_results, summarization_length, claude_model, risk_words):
    risk_summaries = {}
    all_articles = []
    article_counter = 1

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, risk_word in enumerate(risk_words):
        status_text.text(f"Analyzing risk: {risk_word}")
        risk_info = await collect_risk_info(session, talent_name, risk_word, max_results, TAVILY_API_KEY)
        if risk_info:
            for j, result in enumerate(risk_info["results"][:max_results]):
                article = {
                    "index": article_counter,
                    "title": result["title"],
                    "url": result["url"],
                    "risk_word": risk_word
                }
                all_articles.append(article)
                article_counter += 1

                status_text.text(f"Processing article {article['index']}: {article['title']}")
                webpage_content = await get_webpage_content(session, article["url"])
                if webpage_content:
                    summary = await summarize_risk_content(session, webpage_content, risk_word, summarization_length, claude_model, CLAUDE_API_KEY)
                    if summary:
                        article["summary"] = summary
                        article["risk_score"], article["score_details"] = calculate_advanced_risk_score(summary, risk_word)
                        if risk_word not in risk_summaries:
                            risk_summaries[risk_word] = []
                        risk_summaries[risk_word].append(f"[{article['index']}] {summary} (Risk Score: {article['risk_score']:.2f})")
                    else:
                        st.warning(f"Failed to summarize article {article['index']}")
                else:
                    st.warning(f"Failed to fetch content for article {article['index']}")

        progress_bar.progress((i + 1) / len(risk_words))

    status_text.text("Generating final report...")
    risk_report = await generate_risk_report(session, talent_name, risk_summaries, claude_model, CLAUDE_API_KEY, all_articles)
    
    return {
        "talent_name": talent_name,
        "risk_summaries": risk_summaries,
        "risk_report": risk_report,
        "articles": all_articles
    }

def save_result(result):
    if "saved_results" not in st.session_state:
        st.session_state.saved_results = []
    st.session_state.saved_results.append({
        "timestamp": datetime.now().isoformat(),
        "talent_name": result["talent_name"],
        "risk_report": result["risk_report"],
        "articles": result["articles"]
    })

def display_risk_visualization(result):
    st.subheader("炎上リスク可視化")

    # 平均リスクスコアの計算とゲージチャートの表示
    risk_scores = [article['risk_score'] for article in result['articles'] if 'risk_score' in article]
    if risk_scores:
        avg_risk_score = sum(risk_scores) / len(risk_scores)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "平均炎上リスク度"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps' : [
                    {'range': [0, 33], 'color': "green"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': avg_risk_score
                }
            }
        ))
        st.plotly_chart(fig)
    
    # リスクワード別の棒グラフ
    risk_word_scores = {}
    for article in result['articles']:
        if 'risk_score' in article:
            if article['risk_word'] not in risk_word_scores:
                risk_word_scores[article['risk_word']] = []
            risk_word_scores[article['risk_word']].append(article['risk_score'])
    
    if risk_word_scores:
        avg_risk_word_scores = {word: sum(scores) / len(scores) for word, scores in risk_word_scores.items()}
        chart_data = pd.DataFrame({
            'リスクワード': list(avg_risk_word_scores.keys()),
            'スコア': list(avg_risk_word_scores.values())
        })
        
        st.write("リスクワード別平均スコア:")
        st.write(chart_data)
        
        chart = alt.Chart(chart_data).mark_bar().encode(
            x='リスクワード',
            y='スコア',
            color=alt.Color('スコア:Q', scale=alt.Scale(domain=[0, 33, 66, 100], range=['green', 'yellow', 'orange', 'red']))
        ).properties(
            title='リスクワード別平均スコア'
        )
        
        st.altair_chart(chart, use_container_width=True)

    # スコア詳細の表示
    st.subheader("スコア詳細")
    for article in result['articles']:
        if 'score_details' in article:
            st.write(f"Article {article['index']} - {article['title']}:")
            st.write(f"Total Risk Score: {article['risk_score']:.2f}")
            st.write("Score Breakdown:")
            for key, value in article['score_details'].items():
                st.write(f"  {key}: {value:.2f}")
            st.write("---")                

async def main_async():
    st.title("🔥 タレント炎上リスク評価システム")
    st.text("設定のタレント名に名前を入力して「レポート作成」ボタンを押す")
    st.sidebar.title("設定")

    talent_name = st.sidebar.text_input("タレント名", key="talent_name")
    max_results = st.sidebar.slider("リスクワードごとの取得記事数", 1, 10, 5, key="max_results")
    summarization_length = st.sidebar.slider("要約する際の文字数", 100, 500, 200, key="summarization_length")
    claude_model = st.sidebar.selectbox("使用するモデル", ["claude-3-opus-20240229", "claude-3-haiku-20240307","claude-3-5-sonnet-20240620"], index=1, key="claude_model")

    default_risk_words = "政治的思想・発言\n問題・事件・事故・脱税\n薬物・裏社会・暴力\n性蔑視・ジェンダー・LGBT\n熱愛・不倫\n炎上\n宗教\nヌード"
    risk_words_input = st.sidebar.text_area("リスクワード (1行に1つ)", value=default_risk_words, height=200, key="risk_words")
    risk_words = [word.strip() for word in risk_words_input.split('\n') if word.strip()]

    if st.sidebar.button("レポート作成"):
        if not TAVILY_API_KEY:
            st.error("Tavily API key is required.")
        elif not talent_name.strip():
            st.error("タレント名を入力してください。")
        elif not CLAUDE_API_KEY:
            st.error("Claude API key is required.")
        elif not risk_words:
            st.error("少なくとも1つのリスクワードを入力してください。")
        else:
            st.session_state.processing = True
            async with aiohttp.ClientSession() as session:
                with st.spinner("レポート生成中..."):
                    result = await process_talent_risks(
                        session, 
                        TAVILY_API_KEY, 
                        CLAUDE_API_KEY, 
                        talent_name, 
                        max_results, 
                        summarization_length, 
                        claude_model, 
                        risk_words
                    )
                    
                    if result:
                        save_result(result)
                        st.success("レポート生成完了！")
                        
                        st.subheader("総合リスク評価")
                        st.write(result['risk_report'])
                        
                        display_risk_visualization(result)
                        
                        st.subheader("リスクワード別要約")
                        for risk_word, summaries in result['risk_summaries'].items():
                            with st.expander(f"{risk_word}"):
                                st.write("\n".join(summaries))
                        
                        st.subheader("参照記事一覧")
                        for article in result['articles']:
                            if 'risk_score' in article:
                                st.write(f"[{article['index']}] {article['title']} - {article['url']} (Risk: {article['risk_word']}, Score: {article['risk_score']:.2f})")
                            else:
                                st.write(f"[{article['index']}] {article['title']} - {article['url']} (Risk: {article['risk_word']}, Score: N/A)")
                        
                        # レポートのエクスポート
                        report_json = json.dumps(result, ensure_ascii=False, indent=2)
                        st.download_button(
                            label="レポートをJSONでダウンロード",
                            data=report_json,
                            file_name=f"{talent_name}_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
            st.session_state.processing = False

    if "saved_results" in st.session_state and st.session_state.saved_results:
        st.sidebar.subheader("過去の評価結果")
        for i, saved_result in enumerate(reversed(st.session_state.saved_results)):
            if st.sidebar.button(f"{saved_result['talent_name']} - {saved_result['timestamp']}", key=f"history_{i}"):
                st.subheader(f"過去の評価結果: {saved_result['talent_name']}")
                st.write(saved_result['risk_report'])
                st.write("参照記事:")
                for article in saved_result['articles']:
                    if 'risk_score' in article:
                        st.write(f"[{article['index']}] {article['title']} (Risk Score: {article['risk_score']:.2f})")
                    else:
                        st.write(f"[{article['index']}] {article['title']} (Risk Score: N/A)")

def main():
    st.set_page_config(layout="wide", page_title="タレント炎上リスク評価システム", page_icon="🔥")
    asyncio.run(main_async())

if __name__ == "__main__":
    main()