import os
import argparse
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment

OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE   = 16000 
MIC_DURATION  = 10


SOURCE_TEXT = """
Штучний інтелект та машинне навчання сьогодні є одними з найважливіших напрямів 
розвитку сучасних технологій. Ці галузі охоплюють широкий спектр методів, 
алгоритмів та підходів, що дозволяють комп'ютерним системам навчатися на основі 
даних та приймати рішення без явного програмування кожної дії.
"""


def main():
    args = parse_args()

    if args.mic:
        audio_path = record_microphone("mic_record.wav")
    elif args.file and os.path.exists(args.file):
        audio_path = args.file
        print(f"File: {audio_path}")
    else:
        audio_path = generate_audio(SOURCE_TEXT, "source_audio.wav")



def parse_args():
    p = argparse.ArgumentParser(description="Аудіо-аналіз та NLP")
    p.add_argument("--mic", action="store_true", help="Запис з мікрофону")
    p.add_argument("--file", type=str, default=None, help="Шлях до WAV-файлу")
    return p.parse_args()


def record_microphone(fname, duration: int = MIC_DURATION):
    path = os.path.join(OUTPUT_DIR, fname)

    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
        print(f"Recording {duration}s from the microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.record(source, duration=duration)
    
    with open(path, "wb") as f:
        f.write(audio.get_wav_data())
        print(f"Recorded: {path}")
    return path


def generate_audio(text: str, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    try:
        mp3_path = path.replace(".wav", ".mp3")
        tts = gTTS(text=text, lang="uk", slow=False)
        tts.save(mp3_path)
        AudioSegment.from_mp3(mp3_path).export(path, format="wav")
        os.remove(mp3_path)
        return path

    except Exception as e:
        print(f"gTTS: {e}")


if __name__ == "__main__":
    main()
