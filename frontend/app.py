# frontend/app.py

import os
import sys
import airportsdata
# Make project root importable so we can import backend.*
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px

from backend.db import SessionLocal
from backend.queries import get_filtered_df

# ----------------------- Streamlit Page Config -----------------------

st.set_page_config(
    page_title="Airline Delay & Cancellation Dashboard",
    layout="wide",
)


# ----------------------- Helpers & Caching -----------------------


@st.cache_resource
def get_session():
    return SessionLocal()


@st.cache_data
def load_all_data() -> pd.DataFrame:
    """Load all data from the database into a cached DataFrame."""
    with SessionLocal() as session:
        df = get_filtered_df(session)
    return df


def plot_config():
    """
    Plotly config:
    - disable scroll zoom
    - hide Plotly logo
    - keep only: pan, autoscale, PNG export
    """
    return {
        "displaylogo": False,
        "scrollZoom": False,
        "modeBarButtonsToRemove": [
            "zoom2d",
            "zoomIn2d",
            "zoomOut2d",
            "resetScale2d",
            "select2d",
            "lasso2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "toggleSpikelines",
            "zoomInGeo",
            "zoomOutGeo",
            "resetGeo",
            "hoverClosestGeo",
            "orbitRotation",
            "tableRotation",
        ],
    }


# ----------------------- Metric Options -----------------------

metric_options = {
    "Arrival Delay (minutes)": "arr_delay",
    "Carrier Delay (minutes)": "carrier_delay",
    "Weather Delay (minutes)": "weather_delay",
    "NAS Delay (minutes)": "nas_delay",
    "Security Delay (minutes)": "security_delay",
    "Late Aircraft Delay (minutes)": "late_aircraft_delay",
}
metric_label_map = {v: k for k, v in metric_options.items()}

# airport-level metrics
airport_metric_options = {
    "Avg Delay per Flight (minutes)": "delay_per_flight",
    "Cancellation Rate": "cancel_rate",
    "15+ Min Delay Rate": "del15_rate",
    "Total Delay Minutes": "total_delay",
    "Total Flights": "total_flights",
}
airport_metric_label_map = {v: k for k, v in airport_metric_options.items()}


# ----------------------- Visualization Functions -----------------------


def delay_cause_overview(df: pd.DataFrame):
    """Total delay minutes by cause (bar chart)."""
    cause_cols = [
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
    ]
    data = {"cause": [], "delay_minutes": []}
    for col in cause_cols:
        if col in df.columns:
            data["cause"].append(col.replace("_delay", "").upper())
            data["delay_minutes"].append(df[col].sum())

    summary = pd.DataFrame(data)
    if summary.empty:
        return None

    fig = px.bar(
        summary,
        x="cause",
        y="delay_minutes",
        title="Total Delay Minutes by Cause",
        labels={"cause": "Delay Cause", "delay_minutes": "Total Delay Minutes"},
    )
    return fig


def airline_comparison(df: pd.DataFrame, metric: str = "arr_delay"):
    """Average delay metric by carrier."""
    if df.empty or metric not in df.columns:
        return None

    grouped = (
        df.groupby(["carrier", "carrier_name"], as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
    )
    fig = px.bar(
        grouped,
        x="carrier",
        y=metric,
        hover_data=["carrier_name"],
        title=f"Average {metric} by Carrier",
        labels={"carrier": "Carrier", metric: "Average Delay (minutes)"},
    )
    return fig


def cause_analysis_by_time(df: pd.DataFrame, metric: str):
    """
    Time-series analysis for a single delay cause over time.
    Uses year+month as the time axis.
    """
    if df.empty or metric not in df.columns:
        return None

    df_time = df.copy()
    df_time["date"] = pd.to_datetime(
        df_time["year"].astype(str) + "-" + df_time["month"].astype(str) + "-01"
    )

    grouped = (
        df_time.groupby("date", as_index=False)[metric]
        .sum()
        .sort_values("date")
    )

    if grouped.empty:
        return None

    fig = px.line(
        grouped,
        x="date",
        y=metric,
        markers=True,
        title=f"{metric_label_map.get(metric, metric)} over Time",
        labels={"date": "Date", metric: "Delay (minutes)"},
    )
    return fig


# -------- Airport aggregation (reused by ranking/comparison/maps) --------


def build_airport_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate airport-level metrics once and reuse.
    """
    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby("airport", as_index=False)
        .agg(
            airport_name=("airport_name", "first"),
            total_delay=("arr_delay", "sum"),
            total_flights=("arr_flights", "sum"),
            total_cancelled=("arr_cancelled", "sum"),
            total_del15=("arr_del15", "sum"),
        )
    )

    # derived metrics
    agg["delay_per_flight"] = agg.apply(
        lambda row: (row["total_delay"] / row["total_flights"])
        if row["total_flights"] and row["total_flights"] > 0
        else 0,
        axis=1,
    )
    agg["cancel_rate"] = agg.apply(
        lambda row: (row["total_cancelled"] / row["total_flights"])
        if row["total_flights"] and row["total_flights"] > 0
        else 0,
        axis=1,
    )
    agg["del15_rate"] = agg.apply(
        lambda row: (row["total_del15"] / row["total_flights"])
        if row["total_flights"] and row["total_flights"] > 0
        else 0,
        axis=1,
    )

    return agg


def airport_delay_ranking(
    df: pd.DataFrame,
    top_n: int = 20,
    metric: str = "delay_per_flight",
):
    """
    Airports ranked by selected metric.
    Default: delay_per_flight (avg delay per flight).
    Other options: cancel_rate, del15_rate, total_delay, total_flights.
    """
    agg = build_airport_agg(df)
    if agg.empty or metric not in agg.columns:
        return None

    agg = agg.sort_values(metric, ascending=False).head(top_n)

    # keep our order on x-axis
    cats = pd.unique(agg["airport"])
    agg["airport"] = pd.Categorical(agg["airport"], categories=cats, ordered=True)

    y_label = airport_metric_label_map.get(metric, metric)

    fig = px.bar(
        agg,
        x="airport",
        y=metric,
        hover_data=["airport_name", "total_delay", "total_flights"],
        title=f"Top {top_n} Airports by {y_label}",
        labels={"airport": "Airport", metric: y_label},
    )
    return fig


def airline_multi_metric_comparison(
    df: pd.DataFrame,
    selected_carriers: list[str],
    metrics: list[str],
):
    """
    Compare multiple airlines across multiple metrics.
    Grouped bar: x = metric, color = carrier.
    """
    if df.empty:
        return None

    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return None

    agg = (
        df.groupby(["carrier", "carrier_name"], as_index=False)[metrics]
        .mean()
    )

    if selected_carriers:
        agg = agg[agg["carrier"].isin(selected_carriers)]

    if agg.empty:
        return None

    melted = agg.melt(
        id_vars=["carrier", "carrier_name"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    melted["metric_label"] = melted["metric"].map(
        lambda k: metric_label_map.get(k, k)
    )

    fig = px.bar(
        melted,
        x="metric_label",
        y="value",
        color="carrier",
        barmode="group",
        hover_data=["carrier_name"],
        title="Airline Multi-metric Comparison",
        labels={
            "metric_label": "Metric",
            "value": "Value",
            "carrier": "Carrier",
        },
    )
    return fig


def airport_multi_metric_comparison(
    df: pd.DataFrame,
    selected_airports: list[str],
    metrics: list[str],
):
    """
    Compare multiple airports across multiple metrics.
    Grouped bar: x = metric, color = airport.
    """
    agg = build_airport_agg(df)
    if agg.empty:
        return None

    if selected_airports:
        agg = agg[agg["airport"].isin(selected_airports)]

    # fallback: auto-select top 5 by total_flights
    if agg.empty:
        return None
    if not selected_airports:
        agg = agg.sort_values("total_flights", ascending=False).head(5)

    metrics = [m for m in metrics if m in agg.columns]
    if not metrics:
        return None

    melted = agg.melt(
        id_vars=["airport", "airport_name"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    melted["metric_label"] = melted["metric"].map(
        lambda k: airport_metric_label_map.get(k, k)
    )

    fig = px.bar(
        melted,
        x="metric_label",
        y="value",
        color="airport",
        barmode="group",
        hover_data=["airport_name"],
        title="Airport Multi-metric Comparison",
        labels={
            "metric_label": "Metric",
            "value": "Value",
            "airport": "Airport",
        },
    )
    return fig


def cancellation_bar_chart(df: pd.DataFrame, top_n: int = 20):
    """Bar chart of cancellation rate per airport."""
    agg = build_airport_agg(df)
    if agg.empty:
        return None

    agg = agg.sort_values("cancel_rate", ascending=False).head(top_n)

    fig = px.bar(
        agg,
        x="airport",
        y="cancel_rate",
        hover_data=["airport_name", "total_flights", "total_cancelled"],
        title=f"Top {top_n} Airports by Cancellation Rate",
        labels={"airport": "Airport", "cancel_rate": "Cancellation Rate"},
    )
    return fig


@st.cache_data
def _get_all_airport_coords():
    """Get coordinates for all airports using airportsdata library."""
    airports = airportsdata.load('IATA')
    coords_dict = {}
    for iata, info in airports.items():
        if info.get('lat') and info.get('lon'):
            coords_dict[iata] = (info['lat'], info['lon'])
    return coords_dict


def cancellation_hotspot_map(df: pd.DataFrame):
    """Interactive map of cancellation rate per airport (sample airports)."""
    agg = build_airport_agg(df)
    if agg.empty:
        return None

    coords = _get_all_airport_coords()

    agg["lat"] = agg["airport"].map(lambda c: coords.get(c, (None, None))[0])
    agg["lon"] = agg["airport"].map(lambda c: coords.get(c, (None, None))[1])

    agg_map = agg.dropna(subset=["lat", "lon"])
    if agg_map.empty:
        return None

    agg_map["cancel_rate_pct"] = agg_map["cancel_rate"] * 100

    # Use scatter_mapbox for interactive live map
    fig = px.scatter_mapbox(
        agg_map,
        lat="lat",
        lon="lon",
        color="cancel_rate_pct",
        size="cancel_rate_pct",
        hover_name="airport",
        hover_data={
            "airport_name": True,
            "total_flights": True,
            "total_cancelled": True,
            "cancel_rate_pct": ":.2f",
            "lat": False,
            "lon": False,
        },
        title="Cancellation Hotspot Map",
        labels={"cancel_rate_pct": "Cancellation Rate (%)"},
        zoom=3,
        center={"lat": 39.8283, "lon": -98.5795},
        color_continuous_scale="Reds",
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
    )
    return fig


def delay_hotspot_map(df: pd.DataFrame):
    """Interactive map of average arrival delay per airport."""
    if df.empty or "arr_delay" not in df.columns:
        return None

    agg = (
        df.groupby(["airport", "airport_name"], as_index=False)["arr_delay"]
        .mean()
        .rename(columns={"arr_delay": "avg_arr_delay"})
    )
    coords = _get_all_airport_coords()

    agg["lat"] = agg["airport"].map(lambda c: coords.get(c, (None, None))[0])
    agg["lon"] = agg["airport"].map(lambda c: coords.get(c, (None, None))[1])

    agg_map = agg.dropna(subset=["lat", "lon"])
    if agg_map.empty:
        return None

    # Use scatter_mapbox for interactive live map
    fig = px.scatter_mapbox(
        agg_map,
        lat="lat",
        lon="lon",
        color="avg_arr_delay",
        size="avg_arr_delay",
        hover_name="airport",
        hover_data=["airport_name", "avg_arr_delay"],
        title="Average Arrival Delay Hotspot Map",
        labels={"avg_arr_delay": "Avg Arrival Delay (minutes)"},
        zoom=3,
        center={"lat": 39.8283, "lon": -98.5795},
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
    )
    return fig


# ----------------------- Load Data -----------------------

df_all = load_all_data()

# ----------------------- Sidebar Filters -----------------------

st.sidebar.title("Filters")

years = sorted(df_all["year"].unique())
months = sorted(df_all["month"].unique())
carriers = sorted(df_all["carrier"].unique())
airports = sorted(df_all["airport"].unique())


def reset_filters():
    st.session_state.year_filter = []
    st.session_state.month_filter = []
    st.session_state.carrier_filter = []
    st.session_state.airport_filter = []


# Initialize session_state
if "year_filter" not in st.session_state:
    st.session_state.year_filter = []
if "month_filter" not in st.session_state:
    st.session_state.month_filter = []
if "carrier_filter" not in st.session_state:
    st.session_state.carrier_filter = []
if "airport_filter" not in st.session_state:
    st.session_state.airport_filter = []

selected_years = st.sidebar.multiselect(
    "Year",
    years,
    key="year_filter",
)

selected_months = st.sidebar.multiselect(
    "Month",
    months,
    key="month_filter",
)

selected_carriers = st.sidebar.multiselect(
    "Carrier",
    carriers,
    key="carrier_filter",
)

selected_airports = st.sidebar.multiselect(
    "Airport",
    airports,
    key="airport_filter",
)

st.sidebar.button("Reset Filters", on_click=reset_filters)

# Apply filters
df_filtered = df_all.copy()
if selected_years:
    df_filtered = df_filtered[df_filtered["year"].isin(selected_years)]
if selected_months:
    df_filtered = df_filtered[df_filtered["month"].isin(selected_months)]
if selected_carriers:
    df_filtered = df_filtered[df_filtered["carrier"].isin(selected_carriers)]
if selected_airports:
    df_filtered = df_filtered[df_filtered["airport"].isin(selected_airports)]

st.sidebar.markdown("---")
st.sidebar.write(f"Rows after filtering: **{len(df_filtered):,}**")


# ----------------------- Main Layout -----------------------

st.title("Airline Delay & Cancellation Dashboard")

feature = st.radio(
    "Select Feature",
    [
        "Delay & Cancellation Overview",
        "Delay & Cancellation Hotspot Map",
        "Cause Analysis by Time Period",
        "Interactive Search & Filter",
        "Ranking & Comparison",
    ],
    horizontal=True,
)

# ----------------------- Feature 1: Delay & Cancellation Overview -----------------------
if feature == "Delay & Cancellation Overview":
    st.subheader("Delay & Cancellation Overview")

    if df_filtered.empty:
        st.warning("No data available for the current filters.")
    else:
        col_left, col_right = st.columns(2)

        # ================= LEFT: DELAY OVERVIEW =================
        with col_left:
            st.markdown("### 🕐 Delay Overview")

            cause_cols = [
                "carrier_delay",
                "weather_delay",
                "nas_delay",
                "security_delay",
                "late_aircraft_delay",
            ]
            available_causes = [c for c in cause_cols if c in df_filtered.columns]

            if not available_causes:
                st.warning("No delay cause columns found in the data.")
            else:
                totals = df_filtered[available_causes].sum()

                delay_df = pd.DataFrame(
                    {
                        "Category": [
                            c.replace("_delay", "").replace("_", " ").title()
                            for c in available_causes
                        ],
                        "Value": totals.values,
                    }
                )
                total_delay_sum = delay_df["Value"].sum()
                if total_delay_sum > 0:
                    delay_df["Percent"] = (
                        100 * delay_df["Value"] / total_delay_sum
                    )
                else:
                    delay_df["Percent"] = 0.0

                total_delay_minutes = int(total_delay_sum)
                total_flights = int(df_filtered["arr_flights"].sum())
                total_del15 = int(df_filtered["arr_del15"].sum()) if "arr_del15" in df_filtered.columns else 0

                avg_delay_per_flight = (
                    total_delay_minutes / total_flights
                    if total_flights > 0 else 0.0
                )
                delay_rate_15 = (
                    total_del15 / total_flights
                    if total_flights > 0 else 0.0
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Delay (min)", f"{total_delay_minutes:,}")
                c2.metric("Avg Delay per Flight", f"{avg_delay_per_flight:.2f} min")
                c3.metric("Delayed (15+ min)", f"{delay_rate_15*100:.1f}%")

                fig_delay_pie = px.pie(
                    delay_df,
                    names="Category",
                    values="Percent",
                    hole=0.45,
                    title="Delay Cause Composition (%)",
                )
                st.plotly_chart(
                    fig_delay_pie,
                    use_container_width=True,
                    config=plot_config(),
                )

                fig_delay_bar = px.bar(
                    delay_df.sort_values("Value", ascending=False),
                    x="Category",
                    y="Value",
                    title="Total Delay Minutes by Cause",
                    labels={
                        "Category": "Delay Cause",
                        "Value": "Total Delay (min)",
                    },
                )
                st.plotly_chart(
                    fig_delay_bar,
                    use_container_width=True,
                    config=plot_config(),
                )

                # Summary Table
                st.markdown("#### Delay Cause Snapshot Table")
                st.dataframe(
                    delay_df.sort_values("Percent", ascending=False)
                    .rename(columns={"Value": "Total Delay (min)"}),
                    height=220,
                    use_container_width=True,
                )

        # ================= RIGHT: CANCELLATION OVERVIEW =================
        with col_right:
            st.markdown("### ✈ Cancellation Overview")

            if all(c in df_filtered.columns for c in ["arr_flights", "arr_cancelled"]):
                total_flights = int(df_filtered["arr_flights"].sum())
                total_cancel = int(df_filtered["arr_cancelled"].sum())
                total_del15 = int(df_filtered["arr_del15"].sum()) if "arr_del15" in df_filtered.columns else 0
                total_divert = int(df_filtered["arr_diverted"].sum()) if "arr_diverted" in df_filtered.columns else 0

                on_time = total_flights - total_cancel - total_del15 - total_divert
                if on_time < 0:
                    on_time = 0

                def rate(x: int) -> float:
                    return (x / total_flights) if total_flights > 0 else 0.0

                cancel_rate = rate(total_cancel)
                delay_rate = rate(total_del15)
                on_time_rate = rate(on_time)

                # KPI 카드
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Flights", f"{total_flights:,}")
                c2.metric("Cancelled Flights", f"{total_cancel:,}", f"{cancel_rate*100:.2f}%")
                c3.metric("On-time Rate", f"{on_time_rate*100:.1f}%")

                # Outcome (On-time / Delayed / Cancelled / Diverted)
                status_df = pd.DataFrame(
                    {
                        "Category": [
                            "On-time",
                            "Delayed (15+ min)",
                            "Cancelled",
                            "Diverted",
                        ],
                        "Value": [
                            on_time,
                            total_del15,
                            total_cancel,
                            total_divert,
                        ],
                    }
                )
                status_df = status_df[status_df["Value"] > 0]
                total_status_sum = status_df["Value"].sum()
                if total_status_sum > 0:
                    status_df["Percent"] = (
                        100 * status_df["Value"] / total_status_sum
                    )
                else:
                    status_df["Percent"] = 0.0

                fig_status = px.pie(
                    status_df,
                    names="Category",
                    values="Percent",
                    hole=0.45,
                    title="Flight Outcome Composition (%)",
                )
                st.plotly_chart(
                    fig_status,
                    use_container_width=True,
                    config=plot_config(),
                )

                carrier_cancel = (
                    df_filtered.groupby(["carrier", "carrier_name"], as_index=False)
                    .agg(
                        total_cancelled=("arr_cancelled", "sum"),
                        total_flights=("arr_flights", "sum"),
                    )
                )
                carrier_cancel = carrier_cancel[carrier_cancel["total_flights"] > 0]
                carrier_cancel["cancel_rate"] = (
                    carrier_cancel["total_cancelled"]
                    / carrier_cancel["total_flights"]
                )

                top5_carrier = carrier_cancel.sort_values(
                    "total_cancelled", ascending=False
                ).head(5)

                fig_cancel_bar = px.bar(
                    top5_carrier,
                    x="carrier",
                    y="cancel_rate",
                    hover_data=["carrier_name", "total_cancelled", "total_flights"],
                    title="Cancellation Rate by Carrier (Top 5)",
                    labels={
                        "carrier": "Carrier",
                        "cancel_rate": "Cancellation Rate",
                    },
                )
                st.plotly_chart(
                    fig_cancel_bar,
                    use_container_width=True,
                    config=plot_config(),
                )

                # Summary Table
                st.markdown("#### Cancellation Snapshot Table")
                st.dataframe(
                    status_df.sort_values("Percent", ascending=False)
                    .rename(columns={"Value": "Flights"}),
                    height=220,
                    use_container_width=True,
                )

            else:
                st.warning("Cancellation columns (arr_flights, arr_cancelled) are missing.")

        st.info(
            "This overview dashboard is split into two symmetric panels:\n"
            "- **Left:** Delay causes and how much each contributes to total delay minutes.\n"
            "- **Right:** Overall cancellation and outcome composition for the current filters.\n\n"
            "For **time-based analysis**, use *Cause Analysis by Time Period*.\n"
            "For **airline/airport ranking & comparison**, use *Delay & Cancellation Ranking & Comparison*.\n"
            "For **geographical hotspots**, use *Delay & Cancellation Hotspot Map*."
        )



# ----------------------- Feature 2: Delay Hotspot Analysis -----------------------

elif feature == "Delay & Cancellation Hotspot Map":
    st.subheader("Delay & Cancellation Hotspot Map")

    if df_filtered.empty:
        st.warning("No data available for the current filters.")
    else:
        map_metric = st.radio(
            "Map Metric",
            ["Average Arrival Delay", "Cancellation Rate"],
            horizontal=True,
        )

        if map_metric == "Average Arrival Delay":
            fig = delay_hotspot_map(df_filtered)
        else:
            fig = cancellation_hotspot_map(df_filtered)

        if fig is None:
            st.warning(
                "No airports with mapped coordinates. "
                "Please extend the airport coordinate dictionary if needed."
            )
        else:
            st.plotly_chart(fig, use_container_width=True, config=plot_config())
            
        st.info(
            "This section shows **where** delays and cancellations occur: "
            "geographic distribution of airport performance across the US."
        )


# ----------------------- Feature 3: Cause Analysis by Time Period -----------------------

elif feature == "Cause Analysis by Time Period":
    st.subheader("Cause Analysis by Time Period")

    if df_filtered.empty:
        st.warning("No data available for the current filters.")
    else:
        cause_metric_label = st.selectbox(
            "Delay Cause Metric",
            list(metric_options.keys()),
            index=1,  # e.g., Carrier Delay by default
        )
        cause_metric = metric_options[cause_metric_label]
        
        # Add trend visualization
        df_time = df_filtered.copy()
        df_time["date"] = pd.to_datetime(
            df_time["year"].astype(str) + "-" + df_time["month"].astype(str) + "-01"
        )

        # Single cause over time
        fig = cause_analysis_by_time(df_filtered, metric=cause_metric)
        if fig is None:
            st.warning("Unable to build time-series for the selected cause.")
        else:
            st.plotly_chart(fig, use_container_width=True, config=plot_config())
        
        st.markdown("---")
        st.markdown("### All Delay Causes Over Time")
        
        # All causes over time
        cause_cols = [
            "carrier_delay",
            "weather_delay",
            "nas_delay",
            "security_delay",
            "late_aircraft_delay",
        ]
        available_causes = [c for c in cause_cols if c in df_filtered.columns]
        
        if available_causes:
            trend_data = (
                df_time.groupby("date")[available_causes]
                .sum()
                .reset_index()
                .sort_values("date")
            )
            rename_map = {
                c: c.replace("_delay", "").replace("_", " ").title() + " Delay (min)"
                for c in available_causes
            }
            trend_plot = trend_data.rename(columns=rename_map)

            fig_trend = px.line(
                trend_plot,
                x="date",
                y=list(rename_map.values()),
                markers=True,
                title="Monthly Delay Minutes by Cause",
                labels={"date": "Date"},
            )
            st.plotly_chart(fig_trend, use_container_width=True, config=plot_config())

        st.markdown(
            """
            This chart shows how delay causes evolve over time
            (aggregated by year-month) under the current filters.
            """
        )

# ----------------------- Feature 4: Interactive Search & Filter -----------------------

elif feature == "Interactive Search & Filter":
    st.subheader("Interactive Search & Filter")

    st.markdown(
        """
        - Shows the raw data under the current filters.
        - You can sort columns directly in the table.
        - You can also download the filtered data as CSV for further analysis.
        """
    )

    st.dataframe(df_filtered)

    csv = df_filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_airline_delay.csv",
        mime="text/csv",
    )

# ----------------------- Feature 5: Delay Ranking & Comparison -----------------------

elif feature == "Ranking & Comparison":
    st.subheader("Ranking & Comparison")

    if df_filtered.empty:
        st.warning("No data available for the current filters.")
    else:
        view_mode = st.radio(
            "View Mode",
            ["Ranking", "Comparison"],
            horizontal=True,
        )

        entity = st.radio(
            "Entity",
            ["By Airline", "By Airport"],
            horizontal=True,
        )

        # ---------- RANKING ----------
        if view_mode == "Ranking":

            # Airline ranking
            if entity == "By Airline":
                metric_label = st.selectbox(
                    "Metric",
                    list(metric_options.keys()),
                    index=0,
                )
                metric = metric_options[metric_label]
                fig = airline_comparison(df_filtered, metric=metric)
                if fig is None:
                    st.warning(
                        "Unable to compute airline ranking for the current data."
                    )
                else:
                    st.plotly_chart(fig, use_container_width=True, config=plot_config())

            # Airport ranking
            else:  # By Airport
                top_n = st.slider(
                    "Top N Airports",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                )
                airport_metric_label = st.selectbox(
                    "Airport Metric",
                    list(airport_metric_options.keys()),
                    index=0,  # default: Avg Delay per Flight
                )
                airport_metric = airport_metric_options[airport_metric_label]

                fig = airport_delay_ranking(
                    df_filtered,
                    top_n=top_n,
                    metric=airport_metric,
                )
                if fig is None:
                    st.warning(
                        "Unable to compute airport ranking for the current data."
                    )
                else:
                    st.plotly_chart(fig, use_container_width=True, config=plot_config())

        # ---------- COMPARISON ----------
        else:  # view_mode == "Comparison"

            # Airline comparison (multi-metric, multi-carrier)
            if entity == "By Airline":
                carrier_choices = sorted(df_filtered["carrier"].unique())
                selected_carriers = st.multiselect(
                    "Select Carriers to Compare",
                    carrier_choices,
                )

                metric_labels = st.multiselect(
                    "Metrics",
                    list(metric_options.keys()),
                    default=[
                        "Arrival Delay (minutes)",
                        "Carrier Delay (minutes)",
                    ],
                )
                metric_keys = [metric_options[m] for m in metric_labels]

                fig = airline_multi_metric_comparison(
                    df_filtered,
                    selected_carriers,
                    metric_keys,
                )
                if fig is None:
                    st.warning(
                        "Unable to build airline comparison with current selections."
                    )
                else:
                    st.plotly_chart(fig, use_container_width=True, config=plot_config())

            # Airport comparison (multi-metric, multi-airport)
            else:  # By Airport
                airport_choices = sorted(df_filtered["airport"].unique())
                selected_airports_for_cmp = st.multiselect(
                    "Select Airports to Compare",
                    airport_choices,
                )

                airport_metric_labels = st.multiselect(
                    "Metrics",
                    list(airport_metric_options.keys()),
                    default=[
                        "Avg Delay per Flight (minutes)",
                        "Cancellation Rate",
                    ],
                )
                airport_metric_keys = [
                    airport_metric_options[m] for m in airport_metric_labels
                ]

                fig = airport_multi_metric_comparison(
                    df_filtered,
                    selected_airports_for_cmp,
                    airport_metric_keys,
                )
                if fig is None:
                    st.warning(
                        "Unable to build airport comparison with current selections."
                    )
                else:
                    st.plotly_chart(fig, use_container_width=True, config=plot_config())