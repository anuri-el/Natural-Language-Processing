import os
import argparse
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
from scipy.fft import fft, fftfreq




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
        audio_path = record_microphone("l7_mic_record.wav")
    elif args.file and os.path.exists(args.file):
        audio_path = args.file
        print(f"File: {audio_path}")
    else:
        audio_path = generate_audio(SOURCE_TEXT, "l7_source_audio.wav")

    rate, signal = load_audio(audio_path)
    print(f"Rate: {rate} Hz, duration={len(signal)/rate:.1f}")

    filtered = apply_filters(signal, rate)
    print("Filters applied: spectral subtraction + Butterworth (300-3400 Hz)")

    filt_path = os.path.join(OUTPUT_DIR, "l7_filtered_audio.wav")
    filt_int16 = (np.clip(filtered["normalized"], -1, 1) * 32767).astype(np.int16)
    wavfile.write(filt_path, rate, filt_int16)
    print(f"Filted audio: {filt_path}")



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


def load_audio(path: str):
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if data.max() > 1:
        data /= 32768.0
    return rate, data


def spectral_subtraction(signal: np.ndarray, rate: int, noise_dur: float = 0.2):
    n_noise = int(rate * noise_dur)
    noise_samples = signal[:n_noise]

    frame_len = 512
    hop = frame_len // 2
    window = np.hanning(frame_len)

    noise_frames = []
    for i in range(0, max(1, len(noise_samples) - frame_len), hop):
        frame = noise_samples[i:i+frame_len] * window
        noise_frames.append(np.abs(fft(frame, n=frame_len)))
    noise_mag = np.mean(noise_frames, axis=0) if noise_frames else np.zeros(frame_len)

    output = np.zeros_like(signal)
    for i in range(0, len(signal) - frame_len, hop):
        frame = signal[i:i+frame_len] * window
        spectrum = fft(frame, n=frame_len)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        alpha = 2.0 
        mag_clean = np.maximum(mag - alpha * noise_mag, 0.01 * mag)
        clean_spectrum = mag_clean * np.exp(1j * phase)
        frame_out = np.real(np.fft.ifft(clean_spectrum))[:frame_len]
        output[i:i+frame_len] += frame_out * window

    return output


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


def apply_filters(signal: np.ndarray, rate: int):
    results = {"original": signal.copy()}

    denoised = spectral_subtraction(signal, rate)
    results["denoised"] = denoised

    sos = butter_bandpass(300, 3400, rate, order=4)
    bandpassed = sosfilt(sos, denoised)
    results["bandpassed"] = bandpassed

    max_amp = np.max(np.abs(bandpassed)) + 1e-9
    normalized = bandpassed / max_amp * 0.95
    results["normalized"] = normalized

    return results


if __name__ == "__main__":
    main()
