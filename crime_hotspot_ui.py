import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(layout="wide")
st.title("🚨 Vancouver Crime Hotspot Map")

# Small description with horizontal line and smaller font
st.markdown("""---""", unsafe_allow_html=True)
st.markdown(
    """
    <p style='font-size:14px; color: #555;'>
    This interactive dashboard identifies and visualizes <b>crime hotspots</b> in Vancouver using geospatial clustering.  
    It supports <b>patrol route optimization</b> by analyzing crime intensity, mapping locations to road networks,  
    and rendering heatmaps and clusters using Folium and OpenStreetMap data.
    </p>
    """,
    unsafe_allow_html=True
)

# Map rendering
map_file = "vancouver_crime_hotspots_detailed.html"

if os.path.exists(map_file):
    with open(map_file, "r", encoding="utf-8") as f:
        map_html = f.read()
    components.html(map_html, height=700, width=1200)
else:
    st.error(f"Map file '{map_file}' not found. Please generate it before running this app.")
