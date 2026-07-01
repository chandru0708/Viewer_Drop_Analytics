import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scripts.nlp_engine import get_sentiment, get_top_keywords
from scripts.dl_engine import simple_trend_forecast

from scripts.nlp_engine import get_sentiment, get_top_keywords
from scripts.dl_engine import simple_trend_forecast

app = Flask(__name__)
DATA_PATH = os.path.join("data", "raw", "Master_Entertainment_Data.xlsx")


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/debug-data")
def debug_data():
    try:
        df = load_data()
        return jsonify({
            "status": "success",
            "rows": len(df),
            "columns": df.columns.tolist(),
            "sample": df.head(5).to_dict(orient="records")
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    df.columns = [str(col).strip() for col in df.columns]

    rename_map = {
        "Session Date": "session_date",
        "Viewer ID": "viewer_id",
        "Session ID": "session_id",
        "Title": "content_title",
        "Genre": "genre",
        "Device Type": "device_type",
        "Subscription Plan": "subscription_plan",
        "Content Type": "content_type",
        "Region": "region",
        "Watch Time Minutes": "watch_time_minutes",
        "Content Duration Minutes": "content_duration_minutes",
        "Completion Rate": "completion_rate",
        "Engagement Score": "engagement_score",
        "Is Dropoff": "is_dropoff"
    }

    df = df.rename(columns=rename_map)

    if "session_date" not in df.columns:
        raise ValueError(f"'session_date' missing after rename. Columns found: {df.columns.tolist()}")

    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce")
    df = df.dropna(subset=["session_date"]).copy()

    if df.empty:
        raise ValueError("No valid rows remain after converting session_date.")

    text_defaults = {
        "viewer_id": "Unknown",
        "session_id": "Unknown",
        "content_title": "Unknown Title",
        "genre": "Unknown",
        "device_type": "Unknown",
        "subscription_plan": "Unknown",
        "content_type": "Unknown",
        "region": "Unknown"
    }

    for col, default in text_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)

    numeric_defaults = {
        "watch_time_minutes": 0,
        "content_duration_minutes": 60,
        "completion_rate": np.nan,
        "engagement_score": 50,
        "is_dropoff": 0
    }

    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default

    df["watch_time_minutes"] = pd.to_numeric(df["watch_time_minutes"], errors="coerce").fillna(0)
    df["content_duration_minutes"] = pd.to_numeric(df["content_duration_minutes"], errors="coerce").replace(0, 60).fillna(60)
    df["engagement_score"] = pd.to_numeric(df["engagement_score"], errors="coerce").fillna(50)
    df["is_dropoff"] = pd.to_numeric(df["is_dropoff"], errors="coerce").fillna(0).astype(int)

    df["completion_rate"] = pd.to_numeric(df["completion_rate"], errors="coerce")
    df["completion_rate"] = df["completion_rate"].fillna(
        (df["watch_time_minutes"] / df["content_duration_minutes"]) * 100
    ).clip(0, 100)

    df["session_month"] = df["session_date"].dt.to_period("M").astype(str)
    return df


def apply_filters(df, filters):
    filtered = df.copy()

    mapping = {
        "genres": "genre",
        "device_types": "device_type",
        "subscription_plans": "subscription_plan",
        "content_types": "content_type",
        "regions": "region"
    }

    for key, col in mapping.items():
        values = filters.get(key) or []
        if values:
            filtered = filtered[filtered[col].isin(values)]

    start_date = filters.get("start_date")
    end_date = filters.get("end_date")

    if start_date:
        filtered = filtered[filtered["session_date"] >= pd.to_datetime(start_date)]

    if end_date:
        filtered = filtered[filtered["session_date"] <= pd.to_datetime(end_date)]

    return filtered


def add_anomaly_flags(df):
    out = df.copy()
    out["anomaly_low_completion"] = (out["completion_rate"] < 20).astype(int)
    out["anomaly_short_watch"] = (out["watch_time_minutes"] < 5).astype(int)
    out["anomaly_high_dropoff_engagement"] = (
        (out["is_dropoff"] == 1) & (out["engagement_score"] > out["engagement_score"].median())
    ).astype(int)
    out["anomaly_flag"] = (
        (out["anomaly_low_completion"] + out["anomaly_short_watch"] + out["anomaly_high_dropoff_engagement"]) > 0
    ).astype(int)
    return out


def compute_kpis(df):
    return {
        "total_sessions": int(df["session_id"].nunique()),
        "total_viewers": int(df["viewer_id"].nunique()),
        "avg_completion_rate": round(float(df["completion_rate"].mean()), 2),
        "dropoff_rate": round(float(df["is_dropoff"].mean() * 100), 2),
        "anomaly_rate": round(float(df["anomaly_flag"].mean() * 100), 2),
        "avg_engagement_score": round(float(df["engagement_score"].mean()), 2)
    }


def monthly_summary(df):
    return (
        df.groupby("session_month", as_index=False)
        .agg(
            sessions=("session_id", "count"),
            dropoff_rate_pct=("is_dropoff", lambda x: round(float(x.mean() * 100), 2)),
            avg_completion_rate=("completion_rate", lambda x: round(float(x.mean()), 2))
        )
        .sort_values("session_month")
    )


def category_summary(df, col, label):
    result = (
        df.groupby(col, as_index=False)
        .agg(
            sessions=("session_id", "count"),
            avg_completion_rate=("completion_rate", "mean"),
            avg_engagement_score=("engagement_score", "mean"),
            dropoff_rate_pct=("is_dropoff", lambda x: float(x.mean() * 100))
        )
        .sort_values("sessions", ascending=False)
    )

    result["avg_completion_rate"] = result["avg_completion_rate"].round(2)
    result["avg_engagement_score"] = result["avg_engagement_score"].round(2)
    result["dropoff_rate_pct"] = result["dropoff_rate_pct"].round(2)
    return result.rename(columns={col: label})


def daily_sessions(df):
    out = df.copy()
    out["session_day"] = pd.to_datetime(out["session_date"]).dt.normalize()
    return (
        out.groupby("session_day", as_index=False)
        .agg(sessions=("session_id", "count"))
        .rename(columns={"session_day": "session_date"})
        .sort_values("session_date")
    )


def forecast_sessions(daily_df, periods=30):
    if len(daily_df) < 7:
        last_date = daily_df["session_date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
        baseline = float(daily_df["sessions"].mean()) if len(daily_df) else 0
        return pd.DataFrame({
            "session_date": future_dates,
            "forecast_sessions": [round(baseline, 2)] * periods,
            "lower_ci": [max(round(baseline * 0.9, 2), 0)] * periods,
            "upper_ci": [round(baseline * 1.1, 2)] * periods
        })

    ts = daily_df.set_index("session_date")["sessions"].asfreq("D", fill_value=0)
    fit = ExponentialSmoothing(ts, trend="add", seasonal=None, initialization_method="estimated").fit()
    forecast = fit.forecast(periods)

    residual_std = float(np.std(ts - ts.mean()))
    vals = np.maximum(forecast.values, 0)

    return pd.DataFrame({
        "session_date": forecast.index,
        "forecast_sessions": np.round(vals, 2),
        "lower_ci": np.round(np.maximum(vals - 1.96 * residual_std, 0), 2),
        "upper_ci": np.round(vals + 1.96 * residual_std, 2)
    })


def compute_viewer_segmentation(df):
    base_date = df["session_date"].max() + pd.Timedelta(days=1)

    viewer = (
        df.groupby("viewer_id", as_index=False)
        .agg(
            recency_days=("session_date", lambda x: (base_date - x.max()).days),
            frequency_sessions=("session_id", "count"),
            monetary_watch_time=("watch_time_minutes", "sum"),
            avg_completion_rate=("completion_rate", "mean"),
            avg_engagement_score=("engagement_score", "mean"),
            dropoff_rate=("is_dropoff", "mean")
        )
    )

    viewer["R_score"] = pd.qcut(
        viewer["recency_days"].rank(method="first", ascending=False),
        4,
        labels=[4, 3, 2, 1]
    ).astype(int)

    viewer["F_score"] = pd.qcut(
        viewer["frequency_sessions"].rank(method="first"),
        4,
        labels=[1, 2, 3, 4]
    ).astype(int)

    viewer["M_score"] = pd.qcut(
        viewer["monetary_watch_time"].rank(method="first"),
        4,
        labels=[1, 2, 3, 4]
    ).astype(int)

    def assign_segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 3 and f >= 3 and m >= 3:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Viewers"
        elif r <= 2 and f >= 3:
            return "At Risk Loyalists"
        elif r == 4 and f <= 2:
            return "New Viewers"
        elif row["avg_completion_rate"] < 50 or row["dropoff_rate"] > 0.5 or row["avg_engagement_score"] < 50:
            return "Low Engagement"
        return "Regular Viewers"

    viewer["segment"] = viewer.apply(assign_segment, axis=1)
    viewer["dropoff_rate"] = (viewer["dropoff_rate"] * 100).round(2)
    viewer["avg_completion_rate"] = viewer["avg_completion_rate"].round(2)
    viewer["avg_engagement_score"] = viewer["avg_engagement_score"].round(2)
    viewer["monetary_watch_time"] = viewer["monetary_watch_time"].round(2)

    segment_summary = (
        viewer.groupby("segment", as_index=False)
        .agg(
            viewers=("viewer_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency_sessions=("frequency_sessions", "mean"),
            avg_watch_time=("monetary_watch_time", "mean"),
            avg_completion_rate=("avg_completion_rate", "mean"),
            avg_engagement_score=("avg_engagement_score", "mean"),
            avg_dropoff_rate=("dropoff_rate", "mean")
        )
    )

    numeric_cols = [
        "avg_recency_days",
        "avg_frequency_sessions",
        "avg_watch_time",
        "avg_completion_rate",
        "avg_engagement_score",
        "avg_dropoff_rate"
    ]
    segment_summary[numeric_cols] = segment_summary[numeric_cols].round(2)

    return viewer, segment_summary


@app.route("/api/filter-options")
def filter_options():
    try:
        df = load_data()
        return jsonify({
            "status": "success",
            "options": {
                "genres": sorted(df["genre"].dropna().unique().tolist()),
                "device_types": sorted(df["device_type"].dropna().unique().tolist()),
                "subscription_plans": sorted(df["subscription_plan"].dropna().unique().tolist()),
                "content_types": sorted(df["content_type"].dropna().unique().tolist()),
                "regions": sorted(df["region"].dropna().unique().tolist()),
                "min_date": df["session_date"].min().strftime("%Y-%m-%d"),
                "max_date": df["session_date"].max().strftime("%Y-%m-%d")
            }
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/run-analysis", methods=["POST"])
def run_analysis():
    try:
        payload = request.get_json(silent=True) or {}
        df = add_anomaly_flags(load_data())
        filtered = apply_filters(df, payload)

        if filtered.empty:
            return jsonify({
                "status": "empty",
                "message": "No data available for the selected filters."
            })

        sample_text = payload.get(
            "sample_text",
            "The content has strong growth and positive engagement, but some users show weak retention."
        )

        daily = daily_sessions(filtered)
        forecast = forecast_sessions(daily, periods=30)
        viewer_table, segment_summary = compute_viewer_segmentation(filtered)
        nlp_result = get_sentiment(sample_text)
        keywords = get_top_keywords(sample_text, top_n=8)

        series_for_dl = daily["sessions"].tolist() if len(daily) else filtered["watch_time_minutes"].tolist()
        dl_result = simple_trend_forecast(series_for_dl, future_steps=7)

        return jsonify({
            "status": "success",
            "kpis": compute_kpis(filtered),
            "monthly": monthly_summary(filtered).to_dict(orient="records"),
            "genre_summary": category_summary(filtered, "genre", "genre").to_dict(orient="records"),
            "device_summary": category_summary(filtered, "device_type", "device_type").to_dict(orient="records"),
            "subscription_summary": category_summary(filtered, "subscription_plan", "subscription_plan").to_dict(orient="records"),
            "content_type_summary": category_summary(filtered, "content_type", "content_type").to_dict(orient="records"),
            "region_summary": category_summary(filtered, "region", "region").to_dict(orient="records"),
            "daily": [
                {
                    "session_date": pd.to_datetime(r["session_date"]).strftime("%Y-%m-%d"),
                    "sessions": int(r["sessions"])
                }
                for _, r in daily.iterrows()
            ],
            "forecast": [
                {
                    "session_date": pd.to_datetime(r["session_date"]).strftime("%Y-%m-%d"),
                    "forecast_sessions": float(r["forecast_sessions"]),
                    "lower_ci": float(r["lower_ci"]),
                    "upper_ci": float(r["upper_ci"])
                }
                for _, r in forecast.iterrows()
            ],
            "segmentation_sample": viewer_table.head(20).to_dict(orient="records"),
            "segment_summary": segment_summary.to_dict(orient="records"),
            "nlp_result": nlp_result,
            "keywords": keywords,
            "dl_result": dl_result
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/segmentation", methods=["POST"])
def segmentation():
    try:
        payload = request.get_json(silent=True) or {}
        df = add_anomaly_flags(load_data())
        filtered = apply_filters(df, payload)

        if filtered.empty:
            return jsonify({
                "status": "empty",
                "message": "No data available for the selected filters."
            })

        viewer_table, segment_summary = compute_viewer_segmentation(filtered)
        return jsonify({
            "status": "success",
            "segment_summary": segment_summary.to_dict(orient="records"),
            "viewer_segments_sample": viewer_table.head(20).to_dict(orient="records")
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    print("STARTING FLASK APP")
    app.run(debug=True)