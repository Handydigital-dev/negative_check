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
import chardet
from typing import Dict, List, Tuple, Any, Optional

nltk.download('punkt')

load_dotenv()

# 定数の定義
API_ENDPOINTS: Dict[str, str] = {
    "CLAUDE": "https://api.anthropic.com/v1/messages",
}
RETRY_ATTEMPTS: int = 5
MAX_CONCURRENT_REQUESTS: int = 3
BASE_DELAY: int = 2
MAX_RETRIES: int = 15
MAX_BACKOFF_DELAY: int = 60
WEBPAGE_TIMEOUT: int = 30
KEYWORDS_FILE: str = "keywords.json"

TAVILY_API_KEY: Optional[str] = os.environ.get("TAVILY_API_KEY")
CLAUDE_API_KEY: Optional[str] = os.environ.get("CLAUDE_API_KEY")

def load_keywords() -> Dict[str, List[str]]:
    """
    キーワード辞書をJSONファイルから読み込む

    Returns:
        Dict[str, List[str]]: キーワード辞書
    """
    if os.path.exists(KEYWORDS_FILE):
        with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_keywords(keywords: Dict[str, List[str]]) -> None:
    """
    キーワード辞書をJSONファイルに保存する

    Args:
        keywords (Dict[str, List[str]]): 保存するキーワード辞書
    """
    with open(KEYWORDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(keywords, f, ensure_ascii=False, indent=2)

KEYWORDS: Dict[str, List[str]] = load_keywords()

# セッション状態の初期化
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'result' not in st.session_state:
    st.session_state.result = None
if 'api_status' not in st.session_state:
    st.session_state.api_status = ""
if 'selected_result' not in st.session_state:
    st.session_state.selected_result = None

def calculate_risk_score(text: str) -> float:
    """
    テキストのリスクスコアを計算する

    Args:
        text (str): 分析対象のテキスト

    Returns:
        float: 計算されたリスクスコア
    """
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    sentiment_score = (1 - sentiment) * 50
    subjectivity_score = subjectivity * 50
    risk_score = (sentiment_score + subjectivity_score) / 2
    
    print(f"Text: {text[:100]}...")
    print(f"Sentiment: {sentiment}, Subjectivity: {subjectivity}")
    print(f"Calculated Risk Score: {risk_score}")
    
    return risk_score

async def get_webpage_content(session: aiohttp.ClientSession, url: str) -> str:
    """
    指定されたURLからウェブページの内容を取得する

    Args:
        session (aiohttp.ClientSession): aiohttp セッション
        url (str): 取得対象のURL

    Returns:
        str: 取得されたウェブページの内容（HTMLタグ除去済み）
    """
    try:
        ua = UserAgent()
        headers = {'User-Agent': ua.random}
        async with session.get(url, headers=headers, timeout=WEBPAGE_TIMEOUT) as response:
            response.raise_for_status()
            content = await response.read()
            encoding = chardet.detect(content)['encoding']
            if encoding is None:
                encoding = 'utf-8'
            html = content.decode(encoding, errors='replace')
            return strip_html_tags(html)
    except asyncio.TimeoutError:
        st.warning(f"Timeout error when fetching {url}")
        return ""
    except aiohttp.ClientError as e:
        st.error(f"Error fetching webpage {url}: {e}")
        return ""
    except UnicodeDecodeError as e:
        st.error(f"Error decoding webpage content from {url}: {e}")
        return ""

def strip_html_tags(html: str) -> str:
    """
    HTMLタグを除去する

    Args:
        html (str): HTMLタグを含む文字列

    Returns:
        str: HTMLタグを除去した文字列
    """
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

async def call_claude_api_with_advanced_backoff(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    """
    高度なバックオフ機能を使用してClaude APIを呼び出す

    Args:
        session (aiohttp.ClientSession): aiohttp セッション
        url (str): APIエンドポイントURL
        payload (Dict[str, Any]): APIリクエストのペイロード
        headers (Dict[str, str]): APIリクエストのヘッダー

    Returns:
        Dict[str, Any]: APIレスポンス
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def api_call():
        async with semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status in [429, 529]:
                            delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF_DELAY)
                            st.session_state.api_status = f"API overloaded or rate limited. Retrying in {delay:.2f} seconds..."
                            await asyncio.sleep(delay)
                            continue
                        response.raise_for_status()
                        st.session_state.api_status = ""
                        return await response.json()
                except aiohttp.ClientResponseError as e:
                    if e.status == 529:
                        delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF_DELAY)
                        st.session_state.api_status = f"API overloaded. Retrying in {delay:.2f} seconds..."
                        await asyncio.sleep(delay)
                    elif attempt == MAX_RETRIES - 1:
                        st.session_state.api_status = "Max retries reached. API call failed."
                        raise
                    else:
                        delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF_DELAY)
                        st.session_state.api_status = f"API call failed. Retrying in {delay:.2f} seconds... Error: {e}"
                        await asyncio.sleep(delay)
                except aiohttp.ClientError as e:
                    if attempt == MAX_RETRIES - 1:
                        st.session_state.api_status = "Max retries reached. API call failed."
                        raise
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF_DELAY)
                    st.session_state.api_status = f"API call failed. Retrying in {delay:.2f} seconds... Error: {e}"
                    await asyncio.sleep(delay)
            st.session_state.api_status = "Max retries reached. API call failed."
            raise Exception("Max retries reached")

    return await api_call()

async def collect_risk_info(session: aiohttp.ClientSession, talent_name: str, risk_word: str, max_results: int, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Tavily APIを使用してリスク情報を収集する

    Args:
        session (aiohttp.ClientSession): aiohttp セッション
        talent_name (str): タレント名
        risk_word (str): リスクワード
        max_results (int): 取得する最大結果数
        api_key (str): Tavily API キー

    Returns:
        Optional[Dict[str, Any]]: 収集されたリスク情報、エラー時はNone
    """
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

async def expand_keywords(session: aiohttp.ClientSession, api_key: str, risk_word: str) -> List[str]:
    """
    Claude APIを使用してリスクワードに関連するキーワードを生成する

    Args:
        session (aiohttp.ClientSession): aiohttp セッション
        api_key (str): Claude API キー
        risk_word (str): 拡張するリスクワード

    Returns:
        List[str]: 生成された関連キーワードのリスト
    """
    url = API_ENDPOINTS["CLAUDE"]
    prompt = f"'{risk_word}'に関連する単語を10個、日本語で挙げてください。これらの単語は、タレントの炎上リスクを評価する際に役立つものとします。単語のみをカンマ区切りのリストで出力してください。"
    
    payload = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 150,
        "temperature": 0.5,
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
            related_words = response_data['content'][0]['text'].split(',')
            return [word.strip() for word in related_words]
        else:
            st.error(f"Unexpected response format: {response_data}")
            return []
    except Exception as e:
        st.error(f"Failed to expand keywords: {e}")
        return []

def update_keywords(risk_word: str, related_words: List[str]) -> None:
    """
    キーワード辞書を更新し、新しいカテゴリと関連ワードを追加する

    Args:
        risk_word (str): 新しいリスクワード（カテゴリ）
        related_words (List[str]): 関連ワードのリスト
    """
    global KEYWORDS
    if risk_word not in KEYWORDS:
        KEYWORDS[risk_word] = related_words
        save_keywords(KEYWORDS)
        st.info(f"Added new category '{risk_word}' to KEYWORDS with related words: {', '.join(related_words)}")

async def calculate_advanced_risk_score(session: aiohttp.ClientSession, api_key: str, text: str, risk_category: str) -> Tuple[float, Dict[str, float]]:
    """
    テキストの高度なリスクスコアを計算する

    Args:
        session (aiohttp.ClientSession): aiohttp セッション
        api_key (str): Claude API キー
        text (str): 分析対象のテキスト
        risk_category (str): リスクカテゴリ

    Returns:
        Tuple[float, Dict[str, float]]: 正規化されたリスクスコアとスコアの詳細
    """
    global KEYWORDS
    if risk_category not in KEYWORDS:
        related_words = await expand_keywords(session, api_key, risk_category)
        update_keywords(risk_category, related_words)
    
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    word_counts = Counter(word for word in text.lower().split() if word in KEYWORDS[risk_category])
    keyword_score = sum(word_counts.values()) * 10

    numbers = re.findall(r'\d+', text)
    number_score = sum(int(num) for num in numbers if int(num) > 10) * 0.5

    context_phrases = [
        "謝罪", "批判", "問題", "炎上", "逮捕", "疑惑", "スキャンダル", 
        "降板", "契約解除", "引退", "活動休止", "謹慎", 
        "不倫", "浮気", "ゴシップ", "離婚", "破局", "不祥事", 
        "逮捕状", "摘発", "起訴", "書類送検", "家宅捜索", 
        "薬物", "飲酒運転", "暴行", "窃盗", "脱税", 
        "パワハラ", "セクハラ", "モラハラ", "いじめ", 
        "詐欺", "横領", "違法", "犯罪", "裁判", 
        "暴力団", "反社会的勢力", "違法賭博", 
        "未成年", "淫行", "わいせつ", "児童ポルノ", 
        "暴言", "差別", "炎上商法", "SNS炎上"
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

async def summarize_risk_content(session: aiohttp.ClientSession, content: str, risk_word: str, max_length: int, model: str, api_key: str) -> str:
    """
    リスクコンテンツを要約する

    Args:
        session (aiohttp.ClientSession): aiohttp セッション
        content (str): 要約する内容
        risk_word (str): リスクワード
        max_length (int): 要約の最大長
        model (str): 使用するClaudeモデル
        api_key (str): Claude API キー

    Returns:
        str: 要約されたリスクコンテンツ
    """
    url = API_ENDPOINTS["CLAUDE"]
    prompt = f"以下の文章から{risk_word}に関連する内容を{max_length}文字程度で要約してください。特にネガティブな内容や潜在的なリスクに注目してください。関連する内容がない場合は「関連情報なし」と記載してください。\n\n{content}"
    payload = {
        "model": model,
        "max_tokens": min(max_length, 4096),
        "temperature": 0.5,
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

async def generate_risk_report(session: aiohttp.ClientSession, talent_name: str, risk_summaries: Dict[str, List[str]], model: str, api_key: str, articles: List[Dict[str, Any]]) -> str:
    """
    タレントのリスクレポートを生成する

    Args:
        session (aiohttp.ClientSession): aiohttp セッション
        talent_name (str): タレント名
        risk_summaries (Dict[str, List[str]]): リスク要約
        model (str): 使用するClaudeモデル
        api_key (str): Claude API キー
        articles (List[Dict[str, Any]]): 収集された記事情報

    Returns:
        str: 生成されたリスクレポート
    """
    url = API_ENDPOINTS["CLAUDE"]
    summaries_text = "\n\n".join([f"{word}:\n{summary}" for word, summary in risk_summaries.items()])
    articles_info = "\n".join([f"[{a['index']}] {a['title']} - {a['url']} (Risk: {a['risk_word']})" for a in articles])
    prompt = f"""以下は{talent_name}に関する各リスクワードの要約です。これらの情報を基に、{talent_name}の炎上リスクを評価し、500文字程度のレポートを作成してください。
レポートには全体的なリスク評価と、特に注意すべき点を含めてください。また、重要な情報の出典として、適切な記事番号（[index]の形式）を括弧書きで示してください。

要約:
{summaries_text}

記事一覧:
{articles_info}
"""
    
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0.5,
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

async def process_talent_risks(session: aiohttp.ClientSession, TAVILY_API_KEY: str, CLAUDE_API_KEY: str, talent_name: str, max_results: int, summarization_length: int, claude_model: str, risk_words: List[str]) -> Dict[str, Any]:
    """
    タレントのリスク情報を処理する

    Args:
        session (aiohttp.ClientSession): aiohttp セッション
        TAVILY_API_KEY (str): Tavily API キー
        CLAUDE_API_KEY (str): Claude API キー
        talent_name (str): タレント名
        max_results (int): リスクワードごとの最大結果数
        summarization_length (int): 要約の長さ
        claude_model (str): 使用するClaudeモデル
        risk_words (List[str]): リスクワードのリスト

    Returns:
        Dict[str, Any]: 処理結果
    """
    risk_summaries = {}
    all_articles = []
    article_counter = 1

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, risk_word in enumerate(risk_words):
        status_text.text(f"Analyzing risk: {risk_word}")
        if risk_word not in KEYWORDS:
            await expand_keywords(session, CLAUDE_API_KEY, risk_word)
        
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
                        article["risk_score"], article["score_details"] = await calculate_advanced_risk_score(session, CLAUDE_API_KEY, summary, risk_word)
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

def save_result(result: Dict[str, Any]) -> None:
    """
    結果をセッション状態に保存する

    Args:
        result (Dict[str, Any]): 保存する結果
    """
    if "saved_results" not in st.session_state:
        st.session_state.saved_results = []
    st.session_state.saved_results.append({
        "timestamp": datetime.now().isoformat(),
        "talent_name": result["talent_name"],
        "risk_report": result["risk_report"],
        "articles": result["articles"],
        "risk_summaries": result["risk_summaries"]
    })

def display_risk_visualization(result: Dict[str, Any]) -> None:
    """
    リスク可視化を表示する

    Args:
        result (Dict[str, Any]): 表示する結果
    """
    st.subheader("炎上リスク可視化")

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

def display_result(result: Dict[str, Any]) -> None:
    """
    結果を表示する

    Args:
        result (Dict[str, Any]): 表示する結果
    """
    if result:
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
        
        report_json = json.dumps(result, ensure_ascii=False, indent=2)
        st.download_button(
            label="レポートをJSONでダウンロード",
            data=report_json,
            file_name=f"{result['talent_name']}_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

async def main_async() -> None:
    """
    非同期メイン関数
    """
    st.title("🔥 タレント炎上リスク評価システムβ版")
    st.text("設定のタレント名に名前を入力して「レポート作成」ボタンを押す")
    st.text("「タレント名　各リスクワード」でWEB上を検索▶ヒットした記事をAIで要約▶要約した記事をベースにレポートを作成")
    st.text("スコアは記事内のネガティブな表現などから算出")
    st.sidebar.title("設定")

    api_status_placeholder = st.empty()

    talent_name = st.sidebar.text_input("タレント名", key="talent_name")
    max_results = st.sidebar.slider("リスクワードごとの取得記事数", 1, 10, 5, key="max_results")
    summarization_length = st.sidebar.slider("要約する際の文字数", 100, 500, 350, key="summarization_length")
    claude_model = st.sidebar.selectbox("使用するモデル", ["claude-3-opus-20240229", "claude-3-haiku-20240307", "claude-3-sonnet-20240229"], index=1, key="claude_model")

    default_risk_words = "政治的発言\n事件\n薬物\n反社\n熱愛\n不倫\n炎上\n宗教"
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
                    api_status_placeholder.text(st.session_state.api_status)
                    try:
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
                        api_status_placeholder.empty()
                        
                        if result:
                            save_result(result)
                            st.session_state.result = result
                            st.session_state.selected_result = None
                            st.success("レポート生成完了！")
                    except Exception as e:
                        st.error(f"レポート生成中にエラーが発生しました: {e}")
                    finally:
                        st.session_state.processing = False

    if st.session_state.get('result'):
        display_result(st.session_state.result)

    if "saved_results" in st.session_state and st.session_state.saved_results:
        st.sidebar.subheader("過去の評価結果")
        for i, saved_result in enumerate(reversed(st.session_state.saved_results)):
            if st.sidebar.button(f"{saved_result['talent_name']} - {saved_result['timestamp']}", key=f"history_{i}"):
                st.session_state.selected_result = saved_result
                st.session_state.result = None

    if st.session_state.selected_result:
        st.subheader(f"過去の評価結果: {st.session_state.selected_result['talent_name']}")
        st.write(st.session_state.selected_result['risk_report'])
        
        st.subheader("リスクワード別要約")
        for risk_word, summaries in st.session_state.selected_result['risk_summaries'].items():
            with st.expander(f"{risk_word}"):
                st.write("\n".join(summaries))
        
        st.subheader("参照記事一覧")
        for article in st.session_state.selected_result['articles']:
            if 'risk_score' in article:
                st.write(f"[{article['index']}] {article['title']} - {article['url']} (Risk: {article['risk_word']}, Score: {article['risk_score']:.2f})")
            else:
                st.write(f"[{article['index']}] {article['title']} - {article['url']} (Risk: {article['risk_word']}, Score: N/A)")

def main() -> None:
    """
    メイン関数：Streamlitアプリケーションのセットアップと実行
    """
    st.set_page_config(layout="wide", page_title="タレント炎上リスク評価システム", page_icon="🔥")
    asyncio.run(main_async())

if __name__ == "__main__":
    main()