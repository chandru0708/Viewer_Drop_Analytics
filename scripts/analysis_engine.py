from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_TABLES = BASE_DIR / "reports" / "tables"
REPORT_FIGURES = BASE_DIR / "reports" / "figures"

REPORT_TABLES.mkdir(parents=True, exist_ok=True)
REPORT_FIGURES.mkdir(parents=True, exist_ok=True)


def load_data():
    fact = pd.read_csv(
        RAW_DIR / "Fact_WatchSessions.csv",
        parse_dates=["session_start_ts", "session_end_ts", "session_date"]
    )
    viewers = pd.read_csv(RAW_DIR / "Dim_Viewer.csv", parse_dates=["signup_date"])
    content = pd.read_csv(RAW_DIR / "Dim_Content.csv")
    dates = pd.read_csv(RAW_DIR / "Dim_Date.csv", parse_dates=["full_date"])
    devices = pd.read_csv(RAW_DIR / "Dim_Device.csv")
    platforms = pd.read_csv(RAW_DIR / "Dim_Platform.csv")
    regions = pd.read_csv(RAW_DIR / "Dim_Region.csv")
    return fact, viewers, content, dates, devices, platforms, regions


def build_master_dataset():
    fact, viewers, content, dates, devices, platforms, regions = load_data()

    df = fact.merge(viewers, on="viewer_id", how="left")
    df = df.merge(content, on="content_id", how="left")
    df = df.merge(dates, on="date_id", how="left")
    df = df.merge(devices, on="device_id", how="left")
    df = df.merge(platforms, on="platform_id", how="left")
    df = df.merge(regions, on="region_id", how="left")

    return df


def apply_filters(
    df,
    genres=None,
    device_types=None,
    subscription_plans=None,
    content_types=None,
    regions=None,
    start_date=None,
    end_date=None
):
    filtered = df.copy()

    if genres:
        filtered = filtered[filtered["genre"].astype(str).isin(genres)]

    if device_types:
        filtered = filtered[filtered["device_type"].astype(str).isin(device_types)]

    if subscription_plans:
        filtered = filtered[filtered["subscription_plan"].astype(str).isin(subscription_plans)]

    if content_types:
        filtered = filtered[filtered["content_type"].astype(str).isin(content_types)]

    if regions and "region_name" in filtered.columns:
        filtered = filtered[filtered["region_name"].astype(str).isin(regions)]

    if start_date is not None:
        filtered = filtered[pd.to_datetime(filtered["session_date"]) >= pd.to_datetime(start_date)]

    if end_date is not None:
        filtered = filtered[pd.to_datetime(filtered["session_date"]) <= pd.to_datetime(end_date)]

    return filtered


def compute_kpis(df):
    return {
        "total_sessions": int(len(df)),
        "total_viewers": int(df["viewer_id"].nunique()),
        "total_content_items": int(df["content_id"].nunique()),
        "avg_watch_duration_min": round(float(df["watch_duration_min"].mean()), 2),
        "avg_completion_rate": round(float(df["completion_rate"].mean() * 100), 2),
        "dropoff_rate": round(float(df["dropped_off_flag"].mean() * 100), 2),
        "anomaly_rate": round(float(df["anomaly_flag"].mean() * 100), 2),
        "repeat_watch_rate": round(float(df["repeat_watch_flag"].mean() * 100), 2),
        "avg_engagement_score": round(float(df["engagement_score"].mean()), 2),
    }


def get_monthly_trends(df):
    monthly = df.groupby(pd.to_datetime(df["session_date"]).dt.to_period("M")).agg(
        sessions=("session_id", "count"),
        avg_completion_rate=("completion_rate", "mean"),
        dropoff_rate=("dropped_off_flag", "mean")
    ).reset_index()

    monthly["session_month"] = monthly["session_date"].astype(str)
    monthly["avg_completion_rate_pct"] = (monthly["avg_completion_rate"] * 100).round(2)
    monthly["dropoff_rate_pct"] = (monthly["dropoff_rate"] * 100).round(2)
    return monthly


def get_numeric_summary(df):
    numeric_cols = [
        "watch_duration_min",
        "completion_rate",
        "dropoff_minute",
        "engagement_score",
        "paused_count",
        "rewind_count",
        "fast_forward_count",
        "ad_clicks",
        "watch_hours"
    ]
    available_cols = [c for c in numeric_cols if c in df.columns]
    summary = df[available_cols].describe().transpose().reset_index()
    summary = summary.rename(columns={"index": "metric"})
    return summary


def get_genre_summary(df):
    out = df.groupby("genre").agg(
        sessions=("session_id", "count"),
        avg_completion_rate=("completion_rate", "mean"),
        dropoff_rate=("dropped_off_flag", "mean"),
        avg_engagement_score=("engagement_score", "mean")
    ).reset_index().sort_values("sessions", ascending=False)

    out["avg_completion_rate_pct"] = (out["avg_completion_rate"] * 100).round(2)
    out["dropoff_rate_pct"] = (out["dropoff_rate"] * 100).round(2)
    return out


def get_device_summary(df):
    out = df.groupby("device_type").agg(
        sessions=("session_id", "count"),
        avg_completion_rate=("completion_rate", "mean"),
        dropoff_rate=("dropped_off_flag", "mean")
    ).reset_index()

    out["avg_completion_rate_pct"] = (out["avg_completion_rate"] * 100).round(2)
    out["dropoff_rate_pct"] = (out["dropoff_rate"] * 100).round(2)
    return out


def get_subscription_summary(df):
    out = df.groupby("subscription_plan").agg(
        sessions=("session_id", "count"),
        avg_completion_rate=("completion_rate", "mean"),
        dropoff_rate=("dropped_off_flag", "mean"),
        avg_engagement_score=("engagement_score", "mean")
    ).reset_index()

    out["avg_completion_rate_pct"] = (out["avg_completion_rate"] * 100).round(2)
    out["dropoff_rate_pct"] = (out["dropoff_rate"] * 100).round(2)
    return out


def get_rfm_segments(df):
    snapshot_date = pd.to_datetime(df["session_date"]).max() + pd.Timedelta(days=1)

    rfm = df.groupby("viewer_id").agg(
        Recency=("session_date", lambda x: (snapshot_date - pd.to_datetime(x).max()).days),
        Frequency=("session_id", "count"),
        Monetary=("watch_duration_min", "sum")
    ).reset_index()

    rfm["R_score"] = pd.qcut(rfm["Recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)

    def segment(row):
        total = row["R_score"] + row["F_score"] + row["M_score"]
        if total >= 10:
            return "Champions"
        elif total >= 8:
            return "Loyal"
        elif total >= 6:
            return "Potential Loyalists"
        elif total >= 4:
            return "At Risk"
        return "Hibernating"

    rfm["segment"] = rfm.apply(segment, axis=1)

    counts = rfm["segment"].value_counts().reset_index()
    counts.columns = ["segment", "viewer_count"]

    return rfm, counts


def get_content_performance(df):
    group_cols = ["content_id", "genre", "content_type"]
    if "title" in df.columns:
        group_cols = ["content_id", "title", "genre", "content_type"]

    perf = df.groupby(group_cols).agg(
        sessions=("session_id", "count"),
        unique_viewers=("viewer_id", "nunique"),
        avg_completion_rate=("completion_rate", "mean"),
        dropoff_rate=("dropped_off_flag", "mean"),
        avg_engagement_score=("engagement_score", "mean"),
        repeat_watch_rate=("repeat_watch_flag", "mean")
    ).reset_index().sort_values(["sessions", "avg_completion_rate"], ascending=[False, False])

    perf["avg_completion_rate_pct"] = (perf["avg_completion_rate"] * 100).round(2)
    perf["dropoff_rate_pct"] = (perf["dropoff_rate"] * 100).round(2)
    perf["repeat_watch_rate_pct"] = (perf["repeat_watch_rate"] * 100).round(2)
    return perf


def get_anomaly_analysis(df):
    summary = df.groupby("anomaly_flag").agg(
        sessions=("session_id", "count"),
        avg_watch_duration=("watch_duration_min", "mean"),
        avg_completion_rate=("completion_rate", "mean"),
        avg_ad_clicks=("ad_clicks", "mean")
    ).reset_index()

    flag_columns = [
        "flag_duration_exceeds_content",
        "flag_binge_limit_exceeded",
        "flag_repeat_looping",
        "flag_high_ad_clicks",
        "flag_session_burst"
    ]

    available_flags = [col for col in flag_columns if col in df.columns]

    counts = pd.DataFrame({
        "flag_name": available_flags,
        "count": [int(df[col].sum()) for col in available_flags]
    }).sort_values("count", ascending=False)

    return summary, counts


def get_cohort_retention(df):
    cohort_df = df.copy()
    cohort_df["session_month"] = pd.to_datetime(cohort_df["session_date"]).dt.to_period("M").astype(str)

    first_month = cohort_df.groupby("viewer_id")["session_month"].min().reset_index()
    first_month.columns = ["viewer_id", "cohort_month"]

    cohort_df = cohort_df.merge(first_month, on="viewer_id", how="left")

    cohort_pivot = cohort_df.groupby(["cohort_month", "session_month"])["viewer_id"].nunique().reset_index()
    cohort_pivot = cohort_pivot.pivot(index="cohort_month", columns="session_month", values="viewer_id").fillna(0)

    if cohort_pivot.shape[1] > 0:
        cohort_size = cohort_pivot.iloc[:, 0].replace(0, np.nan)
        retention = cohort_pivot.divide(cohort_size, axis=0).fillna(0).round(3)
    else:
        retention = cohort_pivot.copy()

    return retention


def get_forecast(df):
    daily = df.groupby(pd.to_datetime(df["session_date"])).agg(
        sessions=("session_id", "count")
    ).reset_index().sort_values("session_date")

    daily["rolling_mean_30"] = daily["sessions"].rolling(30, min_periods=1).mean()
    daily["rolling_std_30"] = daily["sessions"].rolling(30, min_periods=1).std().fillna(0)

    last_date = daily["session_date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=90, freq="D")

    base_level = daily["sessions"].tail(30).mean()
    volatility = max(daily["rolling_std_30"].tail(30).mean(), 1)

    forecast = pd.DataFrame({"session_date": future_dates})
    forecast["forecast_sessions"] = np.round(base_level).astype(int)
    forecast["lower_ci"] = np.maximum(
        0, np.round(forecast["forecast_sessions"] - 1.96 * volatility)
    ).astype(int)
    forecast["upper_ci"] = np.round(
        forecast["forecast_sessions"] + 1.96 * volatility
    ).astype(int)

    return daily, forecast


def save_outputs(results):
    if "numeric_summary" in results:
        results["numeric_summary"].to_csv(REPORT_TABLES / "numeric_summary.csv", index=False)

    if "monthly" in results:
        results["monthly"].to_csv(REPORT_TABLES / "monthly_trends.csv", index=False)

    if "genre_summary" in results:
        results["genre_summary"].to_csv(REPORT_TABLES / "genre_summary.csv", index=False)

    if "device_summary" in results:
        results["device_summary"].to_csv(REPORT_TABLES / "device_summary.csv", index=False)

    if "subscription_summary" in results:
        results["subscription_summary"].to_csv(REPORT_TABLES / "subscription_plan_summary.csv", index=False)

    if "rfm_counts" in results:
        results["rfm_counts"].to_csv(REPORT_TABLES / "rfm_segment_counts.csv", index=False)

    if "content_performance" in results:
        results["content_performance"].to_csv(REPORT_TABLES / "content_performance.csv", index=False)

    if "anomaly_counts" in results:
        results["anomaly_counts"].to_csv(REPORT_TABLES / "anomaly_flag_counts.csv", index=False)

    if "forecast" in results:
        results["forecast"].to_csv(REPORT_TABLES / "session_forecast_90_days.csv", index=False)

    if "kpis" in results:
        pd.DataFrame(
            [{"metric": k, "value": v} for k, v in results["kpis"].items()]
        ).to_csv(REPORT_TABLES / "kpi_summary.csv", index=False)

    if "retention" in results and isinstance(results["retention"], pd.DataFrame):
        results["retention"].to_csv(REPORT_TABLES / "cohort_retention_matrix.csv")


def run_analysis(filters=None, persist_outputs=True):
    filters = filters or {}
    df = build_master_dataset()

    filtered_df = apply_filters(
        df,
        genres=filters.get("genres"),
        device_types=filters.get("device_types"),
        subscription_plans=filters.get("subscription_plans"),
        content_types=filters.get("content_types"),
        regions=filters.get("regions"),
        start_date=filters.get("start_date"),
        end_date=filters.get("end_date"),
    )

    if filtered_df.empty:
        return {
            "status": "empty",
            "message": "No data available for the selected filters."
        }

    rfm_df, rfm_counts = get_rfm_segments(filtered_df)
    anomaly_summary, anomaly_counts = get_anomaly_analysis(filtered_df)
    retention = get_cohort_retention(filtered_df)
    daily, forecast = get_forecast(filtered_df)

    results = {
        "status": "success",
        "message": "Analysis completed successfully.",
        "kpis": compute_kpis(filtered_df),
        "monthly": get_monthly_trends(filtered_df),
        "numeric_summary": get_numeric_summary(filtered_df),
        "genre_summary": get_genre_summary(filtered_df),
        "device_summary": get_device_summary(filtered_df),
        "subscription_summary": get_subscription_summary(filtered_df),
        "rfm_counts": rfm_counts,
        "rfm_details": rfm_df,
        "content_performance": get_content_performance(filtered_df),
        "anomaly_summary": anomaly_summary,
        "anomaly_counts": anomaly_counts,
        "retention": retention,
        "daily": daily,
        "forecast": forecast,
        "preview": filtered_df.head(100),
    }

    if persist_outputs:
        save_outputs(results)

    return results