from sqlalchemy import Column, Integer, String, Float
from .db import Base

class FlightDelay(Base):
    __tablename__ = "fact_flight_delay"

    # Composite primary key
    year = Column(Integer, primary_key=True)
    month = Column(Integer, primary_key=True)
    carrier = Column(String(8), primary_key=True)
    airport = Column(String(8), primary_key=True)

    carrier_name = Column(String(64))
    airport_name = Column(String(128))

    arr_flights = Column(Integer)
    arr_del15 = Column(Integer)
    carrier_ct = Column(Integer)
    weather_ct = Column(Integer)
    nas_ct = Column(Integer)
    security_ct = Column(Integer)
    late_aircraft_ct = Column(Integer)

    arr_cancelled = Column(Integer)
    arr_diverted = Column(Integer)

    arr_delay = Column(Float)
    carrier_delay = Column(Float)
    weather_delay = Column(Float)
    nas_delay = Column(Float)
    security_delay = Column(Float)
    late_aircraft_delay = Column(Float)

    def __repr__(self) -> str:
        return (
            f"<FlightDelay(year={self.year}, month={self.month}, "
            f"carrier={self.carrier}, airport={self.airport})>"
        )
