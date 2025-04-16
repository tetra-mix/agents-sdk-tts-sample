import os 
import io
import wave
from agents.voice import TTSModelSettings, OpenAITTSModel
from openai import AsyncOpenAI
from pathlib import Path
from dotenv import load_dotenv

async def main() :
    
    grandparent_dir = Path(__file__).resolve().parents[2]
    dotenv_path = grandparent_dir / '.env'
    load_dotenv(dotenv_path=dotenv_path)

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    custom_tts_settings = TTSModelSettings(
        instructions="あなたは音声案内エージェントです。"
        "音声: 日本語の女性の声で、明るく親しみやすいトーンを持つ。"
        "トーン: 親しみやすく、明瞭で安心感のある話し方で、聞き手が自信を持ち、落ち着ける雰囲気を作る。"
        "発音: はっきりと明瞭に、安定したリズムで話し、自然な会話の流れを保ちながらも指示がわかりやすく伝わるようにする。"
        "テンポ: 比較的速めに話し、質問の前後に短い間を挟む。"
        "感情: 温かく支えとなるような話し方で、共感と気遣いを込め、聞き手が安心してガイドを受けられるようにする。",
        voice="sage"
    )

    tts_model = OpenAITTSModel(
        model="gpt-4o-mini-tts",
        openai_client=AsyncOpenAI(
            api_key=OPENAI_API_KEY
            ),
        )
    text = "こんにちは、私はあなたの音声案内エージェントです。"
    response_chunks = []
    async for chunk in tts_model.run(text, custom_tts_settings):
        response_chunks.append(chunk)
    response_audio = b"".join(response_chunks)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1) # モノラル
        wf.setsampwidth(2) # 16ビットPCM
        wf.setframerate(24000) # 24kHz
        wf.writeframes(response_audio)
    
    wav_buffer.seek(0)
    with open("output.wav", "wb") as f:
        f.write(wav_buffer.read())
    wav_buffer.close()
    print("Audio saved as output.wav")
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())