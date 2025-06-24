import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(layout="wide")
st.title("🚨 Vancouver Crime Hotspot Map")

# File path to the saved HTML map
map_file = "/Users/aryanbansal/Documents/Projects/vancouver_crime_hotspots_detailed.html"

# Check if file exists
if os.path.exists(map_file):
    with open(map_file, "r", encoding="utf-8") as f:
        map_html = f.read()
    components.html(map_html, height=700, width=1200)
else:
    st.error(f"Map file '{map_file}' not found. Please generate it before running this app.")
