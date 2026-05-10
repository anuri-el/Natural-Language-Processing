import os
import math
import spacy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
from collections import Counter
from gtts import gTTS
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, spectrogram
from scipy.fft import fft, fftfreq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE   = 16000 
MIC_DURATION  = 30

nlp = spacy.load("uk_core_news_lg")

SEP= "=" * 67
SOURCE_TEXT = """
Штучний інтелект та машинне навчання сьогодні є одними з найважливіших напрямів 
розвитку сучасних технологій. Ці галузі охоплюють широкий спектр методів, 
алгоритмів та підходів, що дозволяють комп'ютерним системам навчатися на основі 
даних та приймати рішення без явного програмування кожної дії.

Нейронні мережі, натхненні будовою людського мозку, стали основою більшості 
сучасних систем штучного інтелекту. Вони складаються з великої кількості 
взаємопов'язаних вузлів — нейронів, які обробляють інформацію та передають 
сигнали один одному. Глибинне навчання, що використовує багатошарові нейронні 
мережі, дозволило досягти надзвичайних результатів у таких задачах, як 
розпізнавання зображень, обробка природної мови та синтез мовлення.

Обробка природної мови є однією з ключових галузей штучного інтелекту. Вона 
включає в себе різноманітні задачі: автоматичний переклад між мовами, аналіз 
тональності текстів, виявлення іменованих сутностей, побудову систем запитань 
та відповідей, а також генерацію текстів. Такі моделі, як BERT, GPT та їхні 
варіанти, революціонізували підходи до розуміння та генерації природної мови.

Аналіз аудіо-повідомлень є важливим напрямом у сучасній системі обробки 
інформації. Перетворення мовлення на текст та тексту на мовлення відкривають 
нові можливості для взаємодії людини з машиною. Системи розпізнавання мовлення 
сьогодні досягли такого рівня точності, що здатні розуміти мовлення різних 
людей з різними акцентами та в умовах навколишнього шуму.

""".strip()


def main():
    args = parse_args()

    if args.mic:
        audio_path = record_microphone("l7_mic_record.wav")
    elif args.file and os.path.exists(args.file):
        audio_path = args.file
        print(f"File: {audio_path}")
    else:
        audio_path = generate_audio(SOURCE_TEXT, "l7_source_audio.wav")

    print(f"\n{SEP}")
    rate, signal = load_audio(audio_path)
    print(f"Rate: {rate} Hz, duration={len(signal)/rate:.1f}")

    filtered = apply_filters(signal, rate)
    print("Filters applied: spectral subtraction + Butterworth (300-3400 Hz)")

    filt_path = os.path.join(OUTPUT_DIR, "l7_filtered_audio.wav")
    filt_int16 = (np.clip(filtered["normalized"], -1, 1) * 32767).astype(np.int16)
    wavfile.write(filt_path, rate, filt_int16)
    print(f"Filted audio: {filt_path}")

    print(f"\n{SEP}")
    features = extract_audio_features(filtered["normalized"], rate)
    print(f"Duration: {features['duration_sec']}s")
    print(f"rms: {features['rms']}")
    print(f"zcr: {features['zcr']}")
    print(f"Спект. центроїд: {features['spectral_centroid']:.1f} Гц")
    print(f"spectral_bandwidth: {features['spectral_bandwidth']:.1f} Гц")
    print(f"spectral_flatness: {features['spectral_flatness']:.1f} Гц")
    print(f"MFCC (перші 5): {features['mfcc'][:5]}")

    print(f"\n{SEP}")
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


    print(f"\n{SEP}")
    annotation = auto_annotate(nlp)
    print(f"Annotation: «{annotation[:120]}…»")

    verif_path = verify_tts(annotation, "l7_verification_audio.wav")

    plot_audio_analysis(filtered, rate, "l7_audio_analysis.png")
    plot_top_words(nlp, "l7_top_words.png")
    plot_word_length_distribution(nlp, "l7_word_length.png")
    plot_pos_distribution(nlp, "l7_pos_distribution.png")
    plot_top_bigrams(nlp, "l7_top_bigrams.png")


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
        print("STT: speech not recognized")
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


def auto_annotate(nlp):
    sentences = nlp["sentences"]
    sent_clean = nlp["sent_clean"]
    X = nlp["X_tfidf"]

    if X.shape[0] < 2:
        return sentences[0] if sentences else "Анотація недоступна."

    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)

    scores = np.ones(sim.shape[0]) / sim.shape[0]
    d = 0.85
    for _ in range(30):
        row_sums = sim.sum(axis=1, keepdims=True) + 1e-9
        trans = sim / row_sums
        scores = (1-d)/sim.shape[0] + d * trans.T @ scores

    best_idx = int(np.argmax(scores))
    best_sent = sentences[min(best_idx, len(sentences)-1)]
    return best_sent.strip()


def verify_tts(annotation: str, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    try:
        mp3 = path.replace(".wav", ".mp3")
        gTTS(text=annotation, lang="uk").save(mp3)
        return mp3
    except Exception as e:
        print(f"[err] {e}")


def plot_audio_analysis(filtered: dict, rate: int, fname: str):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.4)

    signal_orig = filtered["original"]
    signal_filt = filtered["normalized"]
    t_orig = np.linspace(0, len(signal_orig)/rate, len(signal_orig))

    axes[0].plot(t_orig, signal_orig, color="#1565C0", lw=0.6, alpha=0.85)
    axes[0].set_title("Original Signal")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_orig, signal_filt, color="#2E7D32", lw=0.6, alpha=0.85)
    axes[1].set_title("Filtered Signal (Butterworth filter + Spectral Subtraction)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(alpha=0.3)

    f_sg, t_sg, Sxx = spectrogram(signal_filt, rate, nperseg=256, noverlap=128)
    pcm = axes[2].pcolormesh(t_sg, f_sg, 10*np.log10(np.abs(Sxx)+1e-9), shading="gouraud", cmap="inferno")
    axes[2].set_ylim(0, 4000)
    axes[2].set_title("Spectrogram (0-4000 Hz)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    plt.colorbar(pcm, ax=axes[2], label="Decibel (dB)")

    fig.suptitle("Audio Signal Analysis")
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_top_words(nlp: dict, fname):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    words15 = [w for w, _ in nlp["freq_content"].most_common(15)]
    counts15 = [c for _, c in nlp["freq_content"].most_common(15)]
    colors15 = plt.cm.YlOrRd(np.linspace(0.3, 0.9, 15))
    
    ax.barh(list(reversed(words15)), list(reversed(counts15)), color=list(reversed(colors15)), alpha=0.87)
    ax.set_title("Top-15 terms")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Terms")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_word_length_distribution(nlp: dict, fname):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lengths = nlp["lengths"]
    xs = sorted(lengths.keys())
    ys = [lengths[x] for x in xs]
    
    ax.bar(xs, ys, alpha=0.85, edgecolor="white")
    ax.set_title("Word Length Distribution")
    ax.set_xlabel("Number of letters")
    ax.set_ylabel("Frequency")
        
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_pos_distribution(nlp: dict, fname):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    pos_d = nlp["pos_dist"]
    pos_labels = list(pos_d.keys())
    pos_vals = list(pos_d.values())
    
    wedges, texts, autotexts = ax.pie(pos_vals, labels=pos_labels, autopct="%1.1f%%", startangle=140)
    
    for autotext in autotexts:
        autotext.set_color('white')
    
    ax.set_title("POS-distribution")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_top_bigrams(nlp: dict, fname):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bg_labels = [" ".join([token.text for token in bg]) for bg, _ in nlp["top10_bigrams"][:10]]
    bg_vals = [c for _, c in nlp["top10_bigrams"][:10]]
    bg_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(bg_labels)))
    
    ax.barh(list(reversed(bg_labels)), list(reversed(bg_vals)),
            color=list(reversed(bg_colors)), alpha=0.87)
    ax.set_title("Top-10 bigram")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Bigrams")
    
    for i, (label, val) in enumerate(zip(reversed(bg_labels), reversed(bg_vals))):
        ax.text(val + 0.1, i, str(val), va='center')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    main()
