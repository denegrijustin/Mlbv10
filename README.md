# MLB Dashboard

Stable Streamlit dashboard for MLB team review, recent trends, and optional Statcast analytics.

## Files
- app.py
- mlb_api.py
- data_helpers.py
- charts.py
- formatting.py
- requirements.txt

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Deploy to Streamlit Cloud
1. Unzip the package.
2. Open the `upload_to_github` folder.
3. Upload all files in that folder to the root of your GitHub repo.
4. In Streamlit Cloud, set the main file path to `app.py`.
5. Reboot or redeploy the app.

## Notes
- The app loads even if the live feed is unavailable.
- Statcast tabs are optional. If Baseball Savant does not return data, the rest of the dashboard still works.
