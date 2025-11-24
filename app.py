import streamlit as st
import os
import json
import io
import tempfile
from pydub import AudioSegment
from google.cloud import speech
from google.oauth2 import service_account
from google import genai
from google.genai import types
from openai import OpenAI

# --- 頁面設定 ---
st.set_page_config(page_title="AI 語音淨化器", page_icon="✨")
st.title("✨ AI 語音情緒淨化器")
st.markdown("請點擊下方麥克風錄製一段語音，AI 將自動把負面詞彙變成美好的詞語。")

# --- API 設定與 Client 初始化 ---
def get_secret(key):
    if key in st.secrets:
        return st.secrets[key]
    if os.getenv(key):
        return os.getenv(key)
    return None

try:
    # Google Cloud 憑證處理
    if "google_cloud" in st.secrets:
        creds_dict = dict(st.secrets["google_cloud"])
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        speech_client = speech.SpeechClient(credentials=creds)
    else:
        speech_client = speech.SpeechClient() # 本地開發 fallback

    # 其他 Clients
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
                    "start_time": start_seconds, # 保持 float 方便後續計算
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

# --- 核心介面：錄音功能 ---
audio_input = st.audio_input("點擊麥克風開始錄音")

if audio_input is not None:
    if st.button("🚀 開始淨化轉換", type="primary"):
        status_text = st.empty()
        progress_bar = st.progress(0)

        try:
            # Step 1: 讀取錄音與 ASR
            status_text.text("正在聆聽並識別語音 (ASR)...")
            progress_bar.progress(10)
            
            audio_input.seek(0)
            audio_bytes = audio_input.read() # 讀取原始 bytes 供 ASR 和 pydub 使用
            
            audio = speech.RecognitionAudio(content=audio_bytes)
            
            # 設定為 UNSPECIFIED 讓 Google 自動偵測 WAV 格式
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED, 
                sample_rate_hertz=0, # 設為 0 或刪除該行，讓其自動偵測
                language_code="zh-TW",
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
            )

            operation = speech_client.recognize(config=config, audio=audio)
            
            if not operation.results:
                st.warning("沒有偵測到清晰的語音，請再試一次。")
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
            
            st.info(f"識別到的內容: {transcript}")
            progress_bar.progress(30)

            # Step 2: LLM 判斷
            status_text.text("AI 正在思考如何讓這句話更美好...")
            
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
                st.success("這句話很棒，沒有發現負面詞彙！")
                progress_bar.progress(100)
                st.stop()
                
            progress_bar.progress(50)

            # Step 3: 匹配時間軸
            timeline_rules = perform_sliding_window_match(asr_words_data, replacement_map)
            
            with st.expander("查看 AI 替換邏輯細節"):
                st.write(timeline_rules)

            # Step 4: TTS 生成 & 內部混音
            status_text.text("正在生成甜美的聲音並進行混音...")
            
            # 使用 pydub 載入原始音訊
            # 注意: st.audio_input 產生的通常是 wav
            try:
                original_audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            except Exception as e:
                # 如果無法識別，嘗試強制指定 wav
                original_audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

            for rule in timeline_rules:
                # 4-1. 生成 TTS
                speed = 1.0
                if rule['speed_prompt'] == 'fast': speed = 1.2
                elif rule['speed_prompt'] == 'slow': speed = 0.8
                
                tts_resp = openai_client.audio.speech.create(
                    model="tts-1", voice="nova", input=rule['replacement'], speed=speed
                )
                
                # 將 TTS mp3 轉為 AudioSegment
                tts_audio = AudioSegment.from_file(io.BytesIO(tts_resp.content), format="mp3")
                
                # 4-2. 計算時間點 (秒轉毫秒)
                start_ms = int(rule['start_time'] * 1000)
                end_ms = int(rule['end_time'] * 1000)
                
                # 4-3. 混音邏輯: 
                # (A) 將原始音訊中「負面詞」的片段靜音
                # (B) 在該位置疊加新的 TTS 音訊
                
                # 為了避免長度改變導致後面的聲音對不上，我們採用「靜音+疊加」的方式
                # 這樣總時長不變
                
                # 製作一段靜音，長度等於原本的負面詞長度
                silence_duration = end_ms - start_ms
                if silence_duration < 0: silence_duration = 0
                silence_segment = AudioSegment.silent(duration=silence_duration)
                
                # 替換原始區段為靜音 (保持長度不變)
                original_audio = original_audio[:start_ms] + silence_segment + original_audio[end_ms:]
                
                # 疊加 TTS (position 設定在開始時間)
                # 注意：如果 TTS 比原詞長，會蓋到後面的字；如果比較短，會有留白。這是正常的。
                original_audio = original_audio.overlay(tts_audio, position=start_ms)

            progress_bar.progress(90)
            status_text.text("處理完成，正在輸出...")

            # 匯出最終檔案
            buffer = io.BytesIO()
            original_audio.export(buffer, format="mp3")
            final_audio_bytes = buffer.getvalue()

            progress_bar.progress(100)
            status_text.text("✨ 完成！")
            st.balloons()
            
            st.subheader("🎧 您的淨化版語音")
            st.audio(final_audio_bytes, format='audio/mpeg')
            
            st.download_button(
                label="下載 MP3",
                data=final_audio_bytes,
                file_name="censored_recording.mp3",
                mime="audio/mpeg"
            )

        except Exception as e:
            st.error(f"發生錯誤: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
