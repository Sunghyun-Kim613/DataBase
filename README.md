# Airline Delay & Cancellation Dashboard

This project is for a CS 554 final project (Milestones 2–4).  
It uses the BTS Airline On-Time Delay Causes dataset (Airline_Delay_Cause.csv) to build
an interactive dashboard for delay causes and cancellation hotspots.

## Tech Stack

- Frontend: Streamlit
- Backend: SQLite via SQLAlchemy
- Data Processing: Pandas
- Visualization: Plotly

## Structure
(Latest python)
```text
airline_delay_project/
├─ backend/
│  ├─ __init__.py
│  ├─ db.py          # DB connection & Base
│  ├─ models.py      # FlightDelay table
│  ├─ load_data.py   # CSV → DB loader
│  └─ queries.py     # Filtered queries returning pandas DataFrame
├─ frontend/
│  └─ app.py         # Streamlit dashboard
├─ data/
│  └─ Airline_Delay_Cause.csv (you add this)
├─ requirements.txt
└─ README.md
```

## How to Run Locally

1. Create venv (optional) and install dependencies:

```bash
pip install -r requirements.txt
```

2. Put your CSV in `data/Airline_Delay_Cause.csv`.

3. Load data into SQLite:

```bash
python -m backend.load_data
```

4. Run the Streamlit app:

```bash
streamlit run frontend/app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## Features

1. Overview: total delay minutes by cause (bar chart)
2. Airline comparison: average delay per airline
3. Airport ranking: average arrival delay per airport
4. Interactive search & filter: table view + CSV download
5. Cancellation hotspot: **can switch between bar chart and map visualization**

You can deploy this on Streamlit Community Cloud by connecting your GitHub repo
and setting `frontend/app.py` as the main file.
