
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(layout="wide")
st.title("🚨 Vancouver Crime Hotspot Map")

# File path to the saved HTML map
map_file = os.path.join(os.path.dirname(__file__), "vancouver_crime_hotspots_detailed.html")

# Show link to open in new tab (optional: adjust path based on deployment)
st.markdown(
    f'[🌐 Open Fullscreen Map in New Tab](./{map_file}){{:target="_blank"}}',
    unsafe_allow_html=True
)

# Show embedded map
if os.path.exists(map_file):
    with open(map_file, "r", encoding="utf-8") as f:
        map_html = f.read()
    components.html(map_html, height=700, width=1200)
else:
    st.error(f"Map file '{map_file}' not found. Please generate it before running this app.")

st.markdown(
    f"""
    <a href="./{map_file}" target="_blank">
        <button style="padding: 10px 20px; font-size:16px;">🌐 Open Fullscreen Hotspot Map</button>
    </a>
    """,
    unsafe_allow_html=True
)


