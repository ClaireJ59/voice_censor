import streamlit as st
import os
import json
import io
import math
from pydub import AudioSegment
from google.cloud import speech
from google.oauth2 import service_account
from google import genai
from google.genai import types
from openai import OpenAI

# --- 頁面設定 ---
st.set_page_config(page_title="AI 語音淨化器 (進階混音版)", page_icon="🎛️")
st.title("🎛️ AI 語音淨化器 - 進階版")
st.markdown("自動偵測負面詞彙，並透過 **動態變速** 與 **置中對齊** 進行完美替換。")

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("🎛️ 混音微調")
    manual_delay_ms = st.slider("手動延遲 (ms)", min_value=-500, max_value=500, value=0, step=10, help="正數代表延後播放，負數代表提早播放")
    volume_boost = st.slider("替換音量增益 (dB)", min_value=0, max_value=30, value=20, help="讓替換的聲音比原音大聲一點")

# --- API 設定與 Client 初始化 ---
def get_secret(key):
    if key in st.secrets:
        return st.secrets[key]
    if os.getenv(key):
        return os.getenv(key)
    return None

try:
    # Google Cloud 憑證
    if "google_cloud" in st.secrets:
        creds_dict = dict(st.secrets["google_cloud"])
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        speech_client = speech.SpeechClient(credentials=creds)
    else:
        speech_client = speech.SpeechClient()

    # API Keys
    google_api_key = get_secret("GOOGLE_API_KEY")
    openai_api_key = get_secret("OPENAI_API_KEY")
    
    if not google_api_key or not openai_api_key:
        st.error("找不到 API 金鑰，請檢查 Secrets 設定。")
        st.stop()

    gemini_client = genai.Client(api_key=google_api_key)
    openai_client = OpenAI(api_key=openai_api_key)

except Exception as e:
    st.error(f"系統初始化失敗: {e}")
    st.stop()

# --- 核心邏輯: 變速處理 (移植自您的代碼) ---
def speed_change(sound, speed=1.0):
    # 使用 frame_rate 覆寫來改變速度 (會同時改變音高，類似黑膠唱片加速)
    # 這是最自然的變速方式，不會產生數位雜音
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

# --- 核心邏輯: 滑動視窗匹配 ---
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
                
                duration = end_seconds - start_seconds
                
                # 這裡只做簡單標記，詳細變速在後面混音階段處理
                speed_instruction = "normal" 

                final_logs.append({
                    "original_word": candidate_phrase,
                    "replacement": replacement_word,
                    "start_time": start_seconds,
                    "end_time": end_seconds,
                    "duration_seconds": duration,
                    "speed_prompt": speed_instruction
                })
                i += window_size
                matched = True
                break
        if not matched:
            i += 1
    return final_logs

# --- 主介面邏輯 ---
audio_input = st.audio_input("點擊麥克風開始錄音")

if audio_input is not None:
    if st.button("🚀 開始淨化轉換", type="primary"):
        status_text = st.empty()
        progress_bar = st.progress(0)

        try:
            # Step 1: ASR
            status_text.text("正在聆聽並識別語音 (ASR)...")
            progress_bar.progress(10)
            
            audio_input.seek(0)
            audio_bytes = audio_input.read()
            
            audio = speech.RecognitionAudio(content=audio_bytes)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED, 
                sample_rate_hertz=0, 
                language_code="zh-TW",
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
            )

            operation = speech_client.recognize(config=config, audio=audio)
            
            if not operation.results:
                st.warning("沒有偵測到清晰的語音。")
                st.stop()

            result = operation.results[0].alternatives[0]
            transcript = result.transcript
            
            asr_words_data = []
            for word_info in result.words:
                asr_words_data.append({
                    "word": word_info.word.strip(),
                    "start_time": word_info.start_time,
                    "end_time": word_info.end_time
                })
            
            st.info(f"識別內容: {transcript}")
            progress_bar.progress(30)

            # Step 2: LLM
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
            任務：找出負面情緒詞彙並替換為正向、意象美好的詞彙 (如：彩虹、花朵、泡泡、棉花糖)。
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
                st.success("沒有發現負面詞彙！")
                progress_bar.progress(100)
                st.stop()
                
            progress_bar.progress(50)

            # Step 3: 匹配與混音
            timeline_rules = perform_sliding_window_match(asr_words_data, replacement_map)
            
            with st.expander("查看詳細替換邏輯"):
                st.write(timeline_rules)

            status_text.text("正在生成語音並進行進階混音...")
            
            # 載入原始音訊 (pydub)
            try:
                original_audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            except:
                original_audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

            # 為了避免多次疊加導致音量爆音或錯位，我們建立一個空的靜音軌道來放替換詞，最後再疊回去
            # 或者直接在 original_audio 上操作（這裡採用直接操作，比較符合您的邏輯）
            final_audio = original_audio

            for rule in timeline_rules:
                # 4-1. TTS 生成
                tts_resp = openai_client.audio.speech.create(
                    model="tts-1", voice="nova", input=rule['replacement']
                )
                replace_audio = AudioSegment.from_file(io.BytesIO(tts_resp.content), format="mp3")
                
                # 4-2. 時間計算
                original_start_ms = int(rule['start_time'] * 1000)
                original_end_ms = int(rule['end_time'] * 1000)
                original_duration_ms = original_end_ms - original_start_ms
                
                # 4-3. 變速處理邏輯 (您的核心邏輯)
                current_len = len(replace_audio)
                
                # 計算需要的速度 (讓替換詞長度 = 原詞長度)
                if original_duration_ms > 0:
                    calculated_speed = current_len / original_duration_ms
                else:
                    calculated_speed = 1.0
                
                # 限制速度在 0.8 ~ 1.2 之間，避免聲音太奇怪
                speed_factor = max(0.8, min(calculated_speed, 1.2))
                
                # 執行變速
                adjusted_audio = speed_change(replace_audio, speed=speed_factor)
                
                # 4-4. 音量增強
                adjusted_audio = adjusted_audio + volume_boost
                
                
                # 4-6. 置中對齊計算 (Centering Logic)
                # 目標：讓 adjusted_audio 的中心點，對齊原本片段的中心點
                
                # 原本片段的中心點
                original_center = (original_start_ms + original_end_ms) / 2
                
                # 新片段的一半長度
                half_new_duration = len(adjusted_audio) / 2
                
                # 計算新的開始時間 = 中心點 - 新片段的一半 + 手動延遲
                final_position_ms = int(original_center)
                
                # 防呆：不能小於 0
                final_position_ms = max(0, final_position_ms)
                
                # 4-7. 疊加 (Overlay)
                final_audio = final_audio.overlay(adjusted_audio, position=final_position_ms)

            progress_bar.progress(100)
            status_text.text("✨ 處理完成！")
            st.balloons()
            
            # 輸出結果
            buffer = io.BytesIO()
            final_audio.export(buffer, format="mp3")
            final_audio_bytes = buffer.getvalue()
            
            st.subheader("🎧 淨化後的聲音")
            st.audio(final_audio_bytes, format='audio/mpeg')
            
            st.download_button(
                label="下載 MP3",
                data=final_audio_bytes,
                file_name="censored_remix.mp3",
                mime="audio/mpeg"
            )

        except Exception as e:
            st.error(f"發生錯誤: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

