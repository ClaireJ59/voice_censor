import streamlit as st
import os
import json
import io
import shutil
import math
from pydub import AudioSegment
from google.cloud import speech
from google.oauth2 import service_account
from google import genai
from google.genai import types
from openai import OpenAI

# --- 強制設定 FFmpeg 路徑 (解決 iOS 轉檔崩潰問題) ---
# Streamlit Cloud (Debian) 的 ffmpeg 通常在 /usr/bin/ffmpeg
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.ffprobe = shutil.which("ffprobe")
else:
    # 備用路徑
    AudioSegment.converter = "/usr/bin/ffmpeg" 

# --- 頁面設定 ---
st.set_page_config(page_title="AI 語音淨化器 (iOS 穩定版)", page_icon="🎤")
st.title("🎤 AI 語音淨化器")
st.markdown("支援 iOS/Android/PC，自動將負面詞彙轉換為美好意象。")

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("🎛️ 混音設定")
    manual_delay_ms = st.slider("手動延遲修正 (ms)", -500, 500, 0, 10)
    volume_boost = st.slider("替換音量 (dB)", 0, 30, 15)

# --- API 初始化 ---
def get_secret(key):
    if key in st.secrets:
        return st.secrets[key]
    if os.getenv(key):
        return os.getenv(key)
    return None

try:
    if "google_cloud" in st.secrets:
        creds_dict = dict(st.secrets["google_cloud"])
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        speech_client = speech.SpeechClient(credentials=creds)
    else:
        speech_client = speech.SpeechClient()

    google_api_key = get_secret("GOOGLE_API_KEY")
    openai_api_key = get_secret("OPENAI_API_KEY")
    
    if not google_api_key or not openai_api_key:
        st.error("金鑰缺失，請檢查 Secrets。")
        st.stop()

    gemini_client = genai.Client(api_key=google_api_key)
    openai_client = OpenAI(api_key=openai_api_key)

except Exception as e:
    st.error(f"初始化錯誤: {e}")
    st.stop()

# --- Helper Functions ---
def speed_change(sound, speed=1.0):
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

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
                final_logs.append({
                    "original_word": candidate_phrase,
                    "replacement": replacement_word,
                    "start_time": start_seconds,
                    "end_time": end_seconds,
                    "duration_seconds": end_seconds - start_seconds,
                    "speed_prompt": "normal"
                })
                i += window_size
                matched = True
                break
        if not matched: i += 1
    return final_logs

# --- 主介面與邏輯 ---
audio_input = st.audio_input("請按麥克風錄音 (iOS 請稍等幾秒上傳)")

if audio_input is not None:
    # 1. 先顯示錄音檔案資訊，確認 App 沒有崩潰
    audio_input.seek(0, os.SEEK_END)
    file_size = audio_input.tell()
    audio_input.seek(0)
    
    st.info(f"✅ 錄音成功！檔案大小: {file_size / 1024:.1f} KB")
    
    if st.button("🚀 開始淨化", type="primary"):
        status = st.status("正在處理中...", expanded=True)
        
        try:
            # --- Step 1: 格式轉換 (最容易出錯的地方) ---
            status.write("🔄 正在轉換音訊格式 (WAV)...")
            raw_bytes = audio_input.read()
            
            try:
                # 嘗試讀取 (自動偵測格式，包含 m4a)
                input_audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
                
                # 強制轉為 Google 喜歡的格式 (Mono, 16kHz) 減輕負載
                input_audio = input_audio.set_channels(1).set_frame_rate(16000)
                
                wav_buffer = io.BytesIO()
                input_audio.export(wav_buffer, format="wav")
                clean_wav_bytes = wav_buffer.getvalue()
                
            except Exception as ffmpeg_err:
                status.update(label="格式轉換失敗", state="error")
                st.error(f"無法讀取錄音檔，可能是 FFmpeg 未安裝或格式不支援。\n詳細錯誤: {ffmpeg_err}")
                st.stop()

            # --- Step 2: ASR ---
            status.write("👂 正在識別語音 (ASR)...")
            audio = speech.RecognitionAudio(content=clean_wav_bytes)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="zh-TW",
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
            )
            operation = speech_client.recognize(config=config, audio=audio)
            
            if not operation.results:
                status.update(label="識別失敗", state="error")
                st.warning("沒有聽清楚您說的話，請再試一次。")
                st.stop()

            transcript = operation.results[0].alternatives[0].transcript
            asr_words_data = []
            for w in operation.results[0].alternatives[0].words:
                asr_words_data.append({
                    "word": w.word.strip(),
                    "start_time": w.start_time,
                    "end_time": w.end_time
                })
            
            status.write(f"📝 識別內容: {transcript}")

            # --- Step 3: LLM ---
            status.write("🤖 AI 正在審查與替換...")
            prompt = f"""
            你是一位專業的情緒詞彙審查與轉換引擎。
            任務：找出負面情緒詞彙並替換為正向、意象美好的詞彙 (如：彩虹、花朵、泡泡、棉花糖)。
            輸入文本: "{transcript}"
            """
            # 簡化 Schema
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"original_word": {"type": "string"}, "replacement_word": {"type": "string"}},
                    "required": ["original_word", "replacement_word"]
                }
            }
            
            llm_res = gemini_client.models.generate_content(
                model='gemini-2.5-flash', contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema)
            )
            censor_list = json.loads(llm_res.text)
            replacement_map = { i['original_word'].strip(): i['replacement_word'] for i in censor_list }
            
            if not replacement_map:
                status.update(label="完成", state="complete")
                st.success("這句話很棒，沒有負面詞彙！")
                st.stop()

            # --- Step 4: 混音 ---
            status.write("🎹 正在合成與混音...")
            timeline = perform_sliding_window_match(asr_words_data, replacement_map)
            final_audio = input_audio # 使用轉檔後的乾淨音訊當基底

            for rule in timeline:
                tts_resp = openai_client.audio.speech.create(
                    model="tts-1", voice="nova", input=rule['replacement']
                )
                rep_audio = AudioSegment.from_file(io.BytesIO(tts_resp.content), format="mp3")
                
                # 時間與變速
                orig_dur = rule['duration_seconds']
                cur_len = len(rep_audio) / 1000.0
                speed = cur_len / orig_dur if orig_dur > 0 else 1.0
                speed = max(0.8, min(speed, 1.2))
                
                adj_audio = speed_change(rep_audio, speed) + volume_boost
                
                # 置中計算
                orig_center_ms = (rule['start_time'] + rule['end_time']) * 1000 / 2
                pos_ms = int(orig_center_ms)
                
                final_audio = final_audio.overlay(adj_audio, position=max(0, pos_ms))

            # --- 輸出 ---
            status.update(label="處理完成！", state="complete")
            out_buffer = io.BytesIO()
            final_audio.export(out_buffer, format="mp3")
            
            st.subheader("🎧 您的淨化版語音")
            st.audio(out_buffer.getvalue(), format='audio/mpeg')
            st.download_button("下載 MP3", out_buffer.getvalue(), "remix.mp3", "audio/mpeg")

        except Exception as e:
            status.update(label="發生錯誤", state="error")
            st.error(f"執行失敗: {str(e)}")
