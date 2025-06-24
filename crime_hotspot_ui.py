import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(layout="wide")
st.title("🚨 Vancouver Crime Hotspot Map")

# Relative path to HTML file (must be in same directory as this script)
map_file = os.path.join(os.path.dirname(__file__), "vancouver_crime_hotspots_detailed.html")

# Show embedded map if the file exists
if os.path.exists(map_file):
    with open(map_file, "r", encoding="utf-8") as f:
        map_html = f.read()
    components.html(map_html, height=700, width=1200)
else:
    st.error("Map file not found. Please generate it before running this app.")
