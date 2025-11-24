import streamlit as st
import os
import json
import io
import requests
import tempfile
from google.cloud import speech
from google.oauth2 import service_account
from google import genai
from google.genai import types
from openai import OpenAI

# --- 頁面設定 ---
st.set_page_config(page_title="AI 語音淨化器", page_icon="✨")
st.title("✨ AI 語音情緒淨化器")
st.markdown("上傳一段語音，AI 將自動識別負面詞彙並替換為美好的詞語。")

# --- 側邊欄：API 設定 (本地測試用 .env，雲端用 st.secrets) ---
# 為了方便部署，我們優先檢查 st.secrets，如果沒有則嘗試環境變數
# 定義一個讀取金鑰的函數，優先查 Secrets，沒有才查系統變數
def get_secret(key):
    if key in st.secrets:
        return st.secrets[key]
    if os.getenv(key):
        return os.getenv(key)
    return None # 或是拋出錯誤

# 獲取金鑰
google_api_key = get_secret("GOOGLE_API_KEY")
openai_api_key = get_secret("OPENAI_API_KEY")

# 檢查是否成功獲取 (這一步很重要，可以避免報出難懂的錯誤)
if not google_api_key:
    st.error("找不到 GOOGLE_API_KEY，請檢查 Secrets 設定。")
    st.stop()

if not openai_api_key:
    st.error("找不到 OPENAI_API_KEY，請檢查 Secrets 設定。")
    st.stop()

# 初始化 Client
gemini_client = genai.Client(api_key=google_api_key)
openai_client = OpenAI(api_key=openai_api_key)

# --- 核心邏輯函數 (快取以提升效能) ---
# 1. 初始化 Google ASR Client
@st.cache_resource
def get_speech_client():
    # 嘗試從 secrets 讀取 Google Cloud JSON 內容
    if "google_cloud" in st.secrets:
        # 將 secrets 轉換為 dict
        creds_dict = dict(st.secrets["google_cloud"])
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        return speech.SpeechClient(credentials=creds)
    else:
        # 本地開發如果設定了環境變數路徑
        return speech.SpeechClient()

# 2. 初始化其他 Clients
try:
    speech_client = get_speech_client()
    gemini_client = genai.Client(api_key=get_secret("GOOGLE_API_KEY"))
    openai_client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
except Exception as e:
    st.error(f"API 初始化失敗，請檢查 Secrets 設定: {e}")
    st.stop()

EXTERNAL_MIX_URL = "https://a67e4a6a0969.ngrok-free.app/mix"

# --- 輔助函數：滑動視窗匹配 ---
def perform_sliding_window_match(asr_words: list, replacement_map: dict) -> list:
    final_logs = []
    i = 0
    n = len(asr_words)
    MAX_WINDOW_SIZE = 5

    while i < n:
        matched = False
        for window_size in range(min(MAX_WINDOW_SIZE, n - i), 0, -1):
            words_slice = asr_words[i : i + window_size]
            candidate_phrase = "".join([w['word'] for w in words_slice])
            
            if candidate_phrase in replacement_map:
                replacement_word = replacement_map[candidate_phrase]
                start_seconds = words_slice[0]['start_time'].total_seconds()
                end_seconds = words_slice[-1]['end_time'].total_seconds()
                
                duration = end_seconds - start_seconds + 1.5
                if duration <= 0: duration = 0.5
                
                speed_instruction = "normal"
                if duration < 0.4: speed_instruction = "fast"
                elif duration > 1.5: speed_instruction = "slow"

                final_logs.append({
                    "original_word": candidate_phrase,
                    "replacement": replacement_word,
                    "start_time": f"{start_seconds}s",
                    "end_time": f"{end_seconds}s",
                    "duration_seconds": duration,
                    "speed_prompt": speed_instruction
                })
                i += window_size
                matched = True
                break
        if not matched:
            i += 1
    return final_logs

# --- 主介面 ---
uploaded_file = st.file_uploader("請選擇音訊檔案 (WAV, MP3, WEBM)", type=["wav", "mp3", "webm", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/audio', start_time=0)
    
    if st.button("🚀 開始轉換", type="primary"):
        status_text = st.empty()
        progress_bar = st.progress(0)

        try:
            # Step 1: 讀取檔案與 ASR
            status_text.text("正在進行語音識別 (ASR)...")
            progress_bar.progress(10)
            
            audio_content = uploaded_file.read()
            audio = speech.RecognitionAudio(content=audio_content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS, # 若檔案格式不同需調整，或使用 ENCODING_UNSPECIFIED
                sample_rate_hertz=48000,
                language_code="zh-TW",
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
            )

            operation = speech_client.recognize(config=config, audio=audio)
            
            if not operation.results:
                st.error("無法識別語音，請確認音訊清晰度。")
                st.stop()

            result = operation.results[0].alternatives[0]
            transcript = result.transcript
            
            # 整理 ASR 數據
            asr_words_data = []
            for word_info in result.words:
                asr_words_data.append({
                    "word": word_info.word.strip(),
                    "start_time": word_info.start_time,
                    "end_time": word_info.end_time
                })
            
            st.info(f"識別文本: {transcript}")
            progress_bar.progress(30)

            # Step 2: LLM 判斷
            status_text.text("AI 正在審查情緒詞彙...")
            
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "original_word": {"type": "string"},
                        "replacement_word": {"type": "string"}
                    },
                    "required": ["original_word", "replacement_word"]
                }
            }
            
            prompt = f"""
            你是一位專業的情緒詞彙審查與轉換引擎。
            任務：找出負面情緒詞彙並替換為正向、意象美好的詞彙 (如：彩虹、花朵、泡泡)。
            輸入文本: "{transcript}"
            """
            
            llm_response = gemini_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                )
            )
            censor_list = json.loads(llm_response.text)
            replacement_map = { item['original_word'].strip(): item['replacement_word'] for item in censor_list }
            
            if not replacement_map:
                st.success("沒有檢測到負面詞彙！")
                st.stop()
                
            progress_bar.progress(50)

            # Step 3: 匹配時間軸
            timeline_rules = perform_sliding_window_match(asr_words_data, replacement_map)
            st.write("替換計劃:", timeline_rules)

            # Step 4: TTS 生成
            status_text.text("正在生成替換音訊 (TTS)...")
            tts_files = {}
            for idx, rule in enumerate(timeline_rules):
                speed = 1.0
                if rule['speed_prompt'] == 'fast': speed = 1.2
                elif rule['speed_prompt'] == 'slow': speed = 0.8
                
                resp = openai_client.audio.speech.create(
                    model="tts-1", voice="nova", input=rule['replacement'], speed=speed
                )
                tts_files[f"replacement_{idx}"] = (f"rep_{idx}.mp3", io.BytesIO(resp.content), "audio/mpeg")
            
            # Padding
            for i in range(5):
                key = f"replacement_{i}"
                if key not in tts_files:
                    tts_files[key] = ('dummy.bin', io.BytesIO(b'dummy'), 'application/octet-stream')
            
            progress_bar.progress(70)

            # Step 5: 混音
            status_text.text("正在進行最終混音...")
            uploaded_file.seek(0)
            
            censor_rules_json = json.dumps([{
                "replacement": r['replacement'],
                "start_time": r['start_time'],
                "end_time": r['end_time']
            } for r in timeline_rules])

            files_to_upload = {
                'original_audio': (uploaded_file.name, uploaded_file, uploaded_file.type),
                **tts_files
            }
            
            mix_response = requests.post(
                EXTERNAL_MIX_URL,
                data={'censor_rules': censor_rules_json},
                files=files_to_upload
            )

            if mix_response.status_code == 200:
                progress_bar.progress(100)
                status_text.text("完成！")
                st.success("轉換成功！")
                
                # 展示與下載
                st.audio(mix_response.content, format='audio/mpeg')
                st.download_button(
                    label="下載處理後的音訊",
                    data=mix_response.content,
                    file_name="censored_audio.mp3",
                    mime="audio/mpeg"
                )
            else:
                st.error(f"混音服務錯誤: {mix_response.text}")

        except Exception as e:
            st.error(f"發生錯誤: {str(e)}")
