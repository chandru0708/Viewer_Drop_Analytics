import numpy as np

def simple_trend_forecast(series, future_steps=7):
    series = np.array(series, dtype=float)

    if len(series) == 0:
        return {
            "trend": "Stable",
            "confidence": 0.0,
            "forecast": []
        }

    recent = series[-5:] if len(series) >= 5 else series
    slope = recent[-1] - recent[0]

    if slope > 0:
        trend = "Bullish"
    elif slope < 0:
        trend = "Bearish"
    else:
        trend = "Stable"

    last_value = float(series[-1])

    if trend == "Bullish":
        forecast = [round(last_value * (1 + 0.01 * (i + 1)), 2) for i in range(future_steps)]
    elif trend == "Bearish":
        forecast = [round(max(0, last_value * (1 - 0.01 * (i + 1))), 2) for i in range(future_steps)]
    else:
        forecast = [round(last_value, 2) for _ in range(future_steps)]

    confidence = round(min(0.99, abs(slope) / (abs(last_value) + 1e-9) + 0.5), 2)

    return {
        "trend": trend,
        "confidence": confidence,
        "forecast": forecast
    }