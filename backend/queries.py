from typing import Iterable, Optional
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import FlightDelay


def get_filtered_df(
    session: Session,
    years: Optional[Iterable[int]] = None,
    months: Optional[Iterable[int]] = None,
    carriers: Optional[Iterable[str]] = None,
    airports: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return a pandas DataFrame with optional filters applied."""
    stmt = select(FlightDelay)

    if years:
        stmt = stmt.where(FlightDelay.year.in_(list(years)))
    if months:
        stmt = stmt.where(FlightDelay.month.in_(list(months)))
    if carriers:
        stmt = stmt.where(FlightDelay.carrier.in_(list(carriers)))
    if airports:
        stmt = stmt.where(FlightDelay.airport.in_(list(airports)))

    rows = session.execute(stmt).scalars().all()

    data = [
        {
            "year": r.year,
            "month": r.month,
            "carrier": r.carrier,
            "carrier_name": r.carrier_name,
            "airport": r.airport,
            "airport_name": r.airport_name,
            "arr_flights": r.arr_flights,
            "arr_del15": r.arr_del15,
            "carrier_ct": r.carrier_ct,
            "weather_ct": r.weather_ct,
            "nas_ct": r.nas_ct,
            "security_ct": r.security_ct,
            "late_aircraft_ct": r.late_aircraft_ct,
            "arr_cancelled": r.arr_cancelled,
            "arr_diverted": r.arr_diverted,
            "arr_delay": r.arr_delay,
            "carrier_delay": r.carrier_delay,
            "weather_delay": r.weather_delay,
            "nas_delay": r.nas_delay,
            "security_delay": r.security_delay,
            "late_aircraft_delay": r.late_aircraft_delay,
        }
        for r in rows
    ]

    df = pd.DataFrame(data)

    if not df.empty:
        df["cancel_rate"] = df.apply(
            lambda row: (row["arr_cancelled"] / row["arr_flights"])
            if row["arr_flights"] and row["arr_flights"] > 0
            else 0,
            axis=1,
        )

    return df
