import pandas as pd
from pathlib import Path

from .db import Base, engine, SessionLocal
from .models import FlightDelay

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Airline_Delay_Cause.csv"


def safe_int(value, default=0):
    """Safely convert value to int, handling NaN and empty values."""
    if pd.isna(value) or value == '':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0):
    """Safely convert value to float, handling NaN and empty values."""
    if pd.isna(value) or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def create_tables():
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def load_csv_to_db(csv_path: Path = DATA_PATH, chunksize: int = 5000):
    """Load CSV data into the fact_flight_delay table in chunks."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Loading data from {csv_path} ...")
    create_tables()

    chunk_iter = pd.read_csv(csv_path, chunksize=chunksize)
    total_rows = 0

    with SessionLocal() as session:
        for chunk in chunk_iter:
            records = []
            for _, row in chunk.iterrows():
                fd = FlightDelay(
                    year=safe_int(row["year"]),
                    month=safe_int(row["month"]),
                    carrier=str(row["carrier"]),
                    airport=str(row["airport"]),
                    carrier_name=str(row.get("carrier_name", "")),
                    airport_name=str(row.get("airport_name", "")),
                    arr_flights=safe_int(row.get("arr_flights", 0)),
                    arr_del15=safe_int(row.get("arr_del15", 0)),
                    carrier_ct=safe_int(row.get("carrier_ct", 0)),
                    weather_ct=safe_int(row.get("weather_ct", 0)),
                    nas_ct=safe_int(row.get("nas_ct", 0)),
                    security_ct=safe_int(row.get("security_ct", 0)),
                    late_aircraft_ct=safe_int(row.get("late_aircraft_ct", 0)),
                    arr_cancelled=safe_int(row.get("arr_cancelled", 0)),
                    arr_diverted=safe_int(row.get("arr_diverted", 0)),
                    arr_delay=safe_float(row.get("arr_delay", 0.0)),
                    carrier_delay=safe_float(row.get("carrier_delay", 0.0)),
                    weather_delay=safe_float(row.get("weather_delay", 0.0)),
                    nas_delay=safe_float(row.get("nas_delay", 0.0)),
                    security_delay=safe_float(row.get("security_delay", 0.0)),
                    late_aircraft_delay=safe_float(row.get("late_aircraft_delay", 0.0)),
                )
                records.append(fd)

            session.bulk_save_objects(records)
            session.commit()
            total_rows += len(records)
            print(f"Inserted {total_rows} rows so far...")

    print(f"Done. Inserted total {total_rows} rows.")


if __name__ == "__main__":
    load_csv_to_db()
