import csv
import numpy as np
import matplotlib.pyplot as plt
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
    words = [word for word, freq in top3]
    freq = [freq for word, freq in top3]
    return words, freq


def build_term_time_series(filename):
    top3_terms, top3_freq = get_top3_terms(filename)

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


def least_squares_trend(y_values):
    n = len(y_values)
    x = np.arange(1, n+1)

    sum_x = np.sum(x)
    sum_y = np.sum(y_values)
    sum_xy = np.sum(x * y_values)
    sum_x2 = np.sum(x**2)

    a = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)
    b = (sum_y - a*sum_x) / n
    trend = a*x + b

    return a, b, trend


def forecast_next_week(a, b, current_length):
    future_x = np.arange(current_length+1, current_length+8)
    forecast = a*future_x + b
    return future_x, forecast


def analyze_freq_sum(filename):
    sums = []
    last_period = None
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            period = row["День"] + "_" + row["Час"]
            if period != last_period:
                sums.append(int(row["Сума частот"]))
                last_period = period
        
    y = np.array(sums)
    a, b, trend = least_squares_trend(y)
    fx, forecast = forecast_next_week(a, b, len(y))
    x = np.arange(1, len(y)+1)

    plt.figure()
    plt.plot(x, y)
    plt.plot(x, trend, label="trend")
    plt.plot(fx, forecast, label="forecast")
    plt.title("Trend and Forecast")
    plt.legend()
    plt.savefig("output/trend_forecast.png")
    plt.show()


def analyze_top3_series(term_series):
    for term, series in term_series.items():
        y = np.array(series)
        a, b, trend = least_squares_trend(y)
        fx, forecast = forecast_next_week(a, b, len(y))
        x = np.arange(1, len(y)+1)

        plt.figure()
        plt.plot(x, y)
        plt.plot(x, trend, label="trend")
        plt.plot(fx, forecast, label="forecast")
        plt.title(f"trend and forecast: {term}")
        plt.legend()
        plt.savefig(f"output/trend_forecast_{term}.png")
        plt.show()