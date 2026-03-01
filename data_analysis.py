import csv
from collections import defaultdict


def get_top3_terms(filename):
    total_freq = defaultdict(int)

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            word = row["Топ 5"]
            freq = int(row["Частота"])
            total_freq[word] += freq
        
    top3 = sorted(total_freq.items(), key=lambda x: x[1], reverse=True)[:3]
    return [word for word, freq in top3]


def build_term_time_series(filename):
    top3_terms = get_top3_terms(filename)

    term_series = {term: [] for term in top3_terms}
    current_day = None
    daily_sum = {term: 0 for term in top3_terms}

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            day = row["День"]

            if current_day is None:
                current_day = day

            if day != current_day:
                for term in top3_terms:
                    term_series[term].append(daily_sum[term])
                    daily_sum[term] = 0
                current_day = day
            
            word = row["Топ 5"]
            freq = int(row["Частота"])

            if word in top3_terms:
                daily_sum[word] += freq

        for term in top3_terms:
            term_series[term].append(daily_sum[term])
    
    return term_series