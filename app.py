import os
import json
import base64
import io
import asyncio
import requests
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response

# 載入環境變數
from dotenv import load_dotenv
load_dotenv()

# --- SDK Imports ---
from google.cloud import speech
from google import genai
from google.genai import types
from openai import OpenAI

app = FastAPI()

# --- 初始化 Clients ---
# 1. Google ASR Client (建議設定 GOOGLE_APPLICATION_CREDENTIALS 環境變數指向 JSON 金鑰檔)
speech_client = speech.SpeechClient()

# 2. Google Gemini Client
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# 3. OpenAI Client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 設定 ---
EXTERNAL_MIX_URL = "https://a67e4a6a0969.ngrok-free.app/mix"  # 您的外部混音服務 URL

# ==========================================
#  Helper Function: 滑動視窗匹配 (對應 n8n 節點 05)
# ==========================================
def perform_sliding_window_match(asr_words: list, replacement_map: dict) -> list:
    """
    將 ASR 的時間戳記與 LLM 的替換詞進行匹配
    """
    final_logs = []
    i = 0
    n = len(asr_words)
    MAX_WINDOW_SIZE = 5

    while i < n:
        matched = False
        # 從最長視窗開始嘗試匹配 (例如: "誤人子弟" -> 4個字)
        for window_size in range(min(MAX_WINDOW_SIZE, n - i), 0, -1):
            words_slice = asr_words[i : i + window_size]
            
            # 組合候選詞 (去除空白)
            candidate_phrase = "".join([w['word'] for w in words_slice])
            
            if candidate_phrase in replacement_map:
                replacement_word = replacement_map[candidate_phrase]
                
                # 提取時間
                start_time_obj = words_slice[0]['start_time'] # timedelta
                end_time_obj = words_slice[-1]['end_time']   # timedelta
                
                start_seconds = start_time_obj.total_seconds()
                end_seconds = end_time_obj.total_seconds()
                
                duration = end_seconds - start_seconds + 1.5
                if duration <= 0: duration = 0.5
                
                # 簡單的語速提示邏輯
                speed_instruction = "Speak normally."
                if duration < 0.4: speed_instruction = "Speak extremely fast."
                elif duration < 0.8: speed_instruction = "Speak quickly."
                elif duration > 1.5: speed_instruction = "Speak very slowly."

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

# ==========================================
#  Main Endpoint: 處理音訊 (對應 n8n 完整流程)
# ==========================================
@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    接收音訊 -> ASR -> LLM Censor -> TTS -> External Mix -> Return Audio
    """
    try:
        print(f"1. 接收檔案: {file.filename}")
        audio_content = await file.read()

        # ---------------------------------------------------------
        # Step 1: Google ASR (對應節點: 02_ASR_語音轉文字)
        # ---------------------------------------------------------
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS, # 或根據輸入動態調整
            sample_rate_hertz=48000, # 根據實際情況調整，或設為 0 (自動)
            language_code="zh-TW",
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        )

        operation = speech_client.recognize(config=config, audio=audio)
        
        if not operation.results:
            return {"error": "No speech detected"}

        result = operation.results[0].alternatives[0]
        transcript = result.transcript
        
        # 整理 ASR 單詞數據 (保留 timedelta 物件以便計算)
        asr_words_data = []
        for word_info in result.words:
            asr_words_data.append({
                "word": word_info.word.strip(),
                "start_time": word_info.start_time,
                "end_time": word_info.end_time
            })

        print(f"2. ASR 完成: {transcript[:20]}...")

        # ---------------------------------------------------------
        # Step 2: Gemini LLM Censor (對應節點: 04_LLM_Censor判斷)
        # ---------------------------------------------------------
        # 定義輸出的 JSON Schema
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
        你的任務是：找出負面情緒詞彙並替換為正向、意象美好的詞彙 (如：彩虹、花朵、泡泡)。
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
        
        # 建立替換地圖 (Map)
        replacement_map = { 
            item['original_word'].strip(): item['replacement_word'] 
            for item in censor_list 
        }

        print(f"3. LLM 替換規則: {len(replacement_map)} 個")

        # ---------------------------------------------------------
        # Step 3: 滑動視窗匹配 (對應節點: 05_規則計算)
        # ---------------------------------------------------------
        timeline_rules = perform_sliding_window_match(asr_words_data, replacement_map)
        
        if not timeline_rules:
            print("無須替換，直接返回原始音訊或是錯誤")
            # 這裡您可以決定是直接回傳原始檔，還是繼續流程但無替換
            # 為了簡化，我們假設繼續流程

        # ---------------------------------------------------------
        # Step 4: OpenAI TTS 生成 (對應節點: 07_TTS_生成替換音訊)
        # ---------------------------------------------------------
        # 這裡我們使用並發 (Async) 或是簡單的迴圈來生成音訊
        tts_files = {} # 用來存儲 replacement_0, replacement_1...

        for idx, rule in enumerate(timeline_rules):
            tts_prompt = rule['replacement']
            # n8n 節點中還有根據 duration 控制語速的 prompt，這裡簡化處理
            # 若要精確控制語速，OpenAI TTS 只能透過 API 的 'speed' 參數 (0.25 - 4.0)
            
            # 簡單計算 OpenAI speed 參數
            base_speed = 1.0
            if "fast" in rule['speed_prompt']: base_speed = 1.2
            elif "slow" in rule['speed_prompt']: base_speed = 0.8

            response = openai_client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=tts_prompt,
                speed=base_speed
            )
            
            # 將音訊存入記憶體 (BytesIO)
            audio_io = io.BytesIO(response.content)
            tts_files[f"replacement_{idx}"] = (f"rep_{idx}.mp3", audio_io, "audio/mpeg")

        # 補滿 dummy 檔案 (對應節點: 09_資料打包2 的 Padding 邏輯)
        # 您的外部服務似乎預期 5 個 slots
        for i in range(5):
            key = f"replacement_{i}"
            if key not in tts_files:
                tts_files[key] = ('dummy.bin', io.BytesIO(b'dummy'), 'application/octet-stream')

        print(f"4. TTS 生成完成，準備混音")

        # ---------------------------------------------------------
        # Step 5: 呼叫外部混音服務 (對應節點: 10_外部混音服務)
        # ---------------------------------------------------------
        # 準備 multipart/form-data
        
        # 1. 規則 JSON 字串
        censor_rules_json = json.dumps([{
            "replacement": r['replacement'],
            "start_time": r['start_time'],
            "end_time": r['end_time']
        } for r in timeline_rules])

        payload = {'censor_rules': censor_rules_json}
        
        # 2. 檔案部分 (原始音訊 + TTS 音訊)
        # 重置原始音訊的指針以供讀取
        file.file.seek(0)
        files_to_upload = {
            'original_audio': (file.filename, file.file, file.content_type),
            **tts_files
        }

        print("5. 發送至外部混音服務...")
        mix_response = requests.post(
            EXTERNAL_MIX_URL,
            data=payload,
            files=files_to_upload
        )

        if mix_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Mixing service error: {mix_response.text}")

        # ---------------------------------------------------------
        # Step 6: 回傳最終音檔 (對應節點: 11_回傳最終音檔)
        # ---------------------------------------------------------
        return Response(
            content=mix_response.content,
            media_type="audio/mpeg", # 或根據混音服務的返回類型調整
            headers={"Content-Disposition": "attachment; filename=final_mix.mp3"}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
