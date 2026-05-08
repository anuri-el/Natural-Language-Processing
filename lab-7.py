import os
import math
import spacy
import argparse
import numpy as np
import speech_recognition as sr
from collections import Counter
from gtts import gTTS
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
from scipy.fft import fft, fftfreq
from sklearn.feature_extraction.text import TfidfVectorizer


OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE   = 16000 
MIC_DURATION  = 10

nlp = spacy.load("uk_core_news_lg")

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

    features = extract_audio_features(filtered["normalized"], rate)
    print(f"Duration: {features['duration_sec']}с")
    print(f"Спект. центроїд: {features['spectral_centroid']:.1f} Гц")
    print(f"MFCC (перші 5): {features['mfcc'][:5]}")

    recognized_text = speech_to_text(audio_path)

    txt_raw = os.path.join(OUTPUT_DIR, "l7_recognized_text_raw.txt")
    with open(txt_raw, "w", encoding="utf-8") as f:
        f.write(recognized_text)
    print(f"Text saved to {txt_raw}")

    nlp = nlp_pipeline(recognized_text)

    print(f"Sentences: {len(nlp['sentences'])}  |  Tokens: {len(nlp['all_tokens'])}")
    print(f"Після видалення стоп-слів: {len(nlp['content_tokens'])} tokens")
    print(f"POS: {dict(nlp['pos_dist'])}")
    print(f"Top-10: {[w for w,_ in nlp['top10_words']]}")

    


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


def extract_audio_features(signal: np.ndarray, rate: int):
    eps = 1e-9
    rms = float(np.sqrt(np.mean(signal**2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(signal)))) / 2)
    energy = float(np.sum(signal**2))

    N = len(signal)
    freqs = fftfreq(N, d=1/rate)
    spec = np.abs(fft(signal))[:N//2]
    freqs = freqs[:N//2]

    centroid = float(np.sum(freqs * spec) / (np.sum(spec) + eps))
    bandwidth = float(np.sqrt(np.sum((freqs - centroid)**2 * spec) / (np.sum(spec)+eps)))
    flatness = float(np.exp(np.mean(np.log(spec + eps))) / (np.mean(spec) + eps))

    n_mels, n_mfcc = 26, 13
    fmin, fmax = 80, min(8000, rate//2)
    mel_min = 2595 * math.log10(1 + fmin/700)
    mel_max = 2595 * math.log10(1 + fmax/700)
    mel_points = np.linspace(mel_min, mel_max, n_mels+2)
    hz_points  = 700 * (10**(mel_points/2595) - 1)
    bin_points = np.floor((N+1) * hz_points / rate).astype(int)
    bin_points = np.clip(bin_points, 0, N//2-1)

    filter_bank = []
    for m in range(1, n_mels+1):
        f_m_minus = bin_points[m-1]; f_m = bin_points[m]; f_m_plus = bin_points[m+1]
        filter_ = np.zeros(N//2)
        for k in range(f_m_minus, f_m):
            if f_m - f_m_minus > 0:
                filter_[k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus - f_m > 0:
                filter_[k] = (f_m_plus - k) / (f_m_plus - f_m)
        filter_bank.append(np.sum(filter_ * spec))

    log_fb = np.log(np.array(filter_bank) + eps)
    mfcc   = np.zeros(n_mfcc)
    for n in range(n_mfcc):
        mfcc[n] = np.sum(log_fb * np.cos(math.pi*n*(np.arange(n_mels)+0.5)/n_mels))

    return {
        "duration_sec": round(len(signal)/rate, 2),
        "rms": round(rms, 6),
        "energy": round(energy, 4),
        "zcr": round(zcr, 4),
        "spectral_centroid": round(centroid, 1),
        "spectral_bandwidth": round(bandwidth, 1),
        "spectral_flatness": round(flatness, 6),
        "mfcc": [round(float(m), 4) for m in mfcc],
        "freq_bins": freqs.tolist()[:200],
        "spectrum": spec.tolist()[:200],
    }


def speech_to_text(audio_path: str, language: str = "uk-UA"):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    try:
        with sr.AudioFile(audio_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        print(" STT: speech not recognized")
        return ""
    except Exception as e:
        print(f"[err] {e}")


def nlp_pipeline(text: str):
    doc = nlp(text)

    sentences = [sent.text.strip() for sent in doc.sents]
    all_tokens = [token.text for token in doc]

    content_tokens = [token for token in doc if not token.is_stop and len(token.text) > 2]

    lemmas = [token.lemma_ for token in content_tokens]
    
    pos_tags = [(token.text, token.pos_) for token in content_tokens]
    pos_dist = Counter(tag for _, tag in pos_tags)
    
    freq_all = Counter(all_tokens)
    freq_content = Counter([token.text for token in content_tokens])
    freq_lemma = Counter(lemmas)
    top10_words = freq_content.most_common(10)

    bigrams = Counter()
    for i in range(len(content_tokens)-1):
        bigrams[(content_tokens[i], content_tokens[i+1])] += 1
    top10_bigrams = bigrams.most_common(10)

    lengths = Counter(len(t) for t in all_tokens)

    sent_clean = [" ".join([token.text for token in sent if not token.is_stop]) for sent in doc.sents]
    sent_clean = [s for s in sent_clean if s.strip()]
    
    if len(sent_clean) < 2:
        sent_clean = [" ".join([token.text for token in content_tokens])]
    
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2), min_df=1)
    X_tfidf = tfidf.fit_transform(sent_clean)
    
    return {
        "text": text,
        "sentences": sentences,
        "all_tokens": all_tokens,
        "content_tokens": [token.text for token in content_tokens],
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "pos_dist": pos_dist,
        "freq_all": freq_all,
        "freq_content": freq_content,
        "freq_lemma": freq_lemma,
        "top10_words": top10_words,
        "top10_bigrams": top10_bigrams,
        "lengths": lengths,
        "tfidf": tfidf,
        "X_tfidf": X_tfidf,
        "sent_clean": sent_clean,
    }


if __name__ == "__main__":
    main()
