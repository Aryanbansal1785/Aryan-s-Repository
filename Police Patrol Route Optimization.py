import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import folium
import osmnx as ox
import networkx as nx

# Load and process data
file_path = "/Users/aryanbansal/Downloads/crimedata_csv_AllNeighbourhoods_2024/crimedata_csv_AllNeighbourhoods_2024.csv"
df = pd.read_csv(file_path)
df = df.dropna(subset=['X', 'Y'])
df.rename(columns={'X': 'Longitude', 'Y': 'Latitude'}, inplace=True)
df['ReportedDate'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']])
df = df[['ReportedDate', 'TYPE', 'Latitude', 'Longitude', 'NEIGHBOURHOOD']]

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:26910")
gdf = gdf.to_crs("EPSG:4326")
gdf['Latitude'] = gdf.geometry.y
gdf['Longitude'] = gdf.geometry.x

# KMeans Clustering
coords = gdf[['Latitude', 'Longitude']]
kmeans = KMeans(n_clusters=10, random_state=42)
gdf['cluster'] = kmeans.fit_predict(coords)
centers = kmeans.cluster_centers_

plt.figure(figsize=(10, 6))
plt.scatter(gdf['Longitude'], gdf['Latitude'], c=gdf['cluster'], cmap='tab10', s=10)
plt.title("Crime Hotspots in Vancouver (KMeans Clustering)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

# Folium Map of Clusters
m = folium.Map(location=[gdf['Latitude'].mean(), gdf['Longitude'].mean()], zoom_start=12)
for _, row in gdf.iterrows():
    cluster_color = "#{:06x}".format((int(row['cluster']) * 123456) % 0xFFFFFF)
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=2,
        color=cluster_color,
        fill=True,
        fill_opacity=0.6
    ).add_to(m)
m.save("vancouver_crime_clusters_map.html")

# Road Network + Connected Graph
print("Loading road network...")
G = ox.graph_from_place('Vancouver, British Columbia, Canada', network_type='drive')

# Get largest connected component
print("Finding largest connected component...")
if not nx.is_strongly_connected(G):
    # For directed graphs, use strongly connected components
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Map cluster centers to nearest nodes in the connected component
print("Mapping hotspots to road network...")
hotspot_nodes = []
valid_centers = []
hotspot_info = []

for i, center in enumerate(centers):
    try:
        # Find nearest node (lon, lat order for osmnx)
        node = ox.distance.nearest_nodes(G, center[1], center[0])
        
        # Verify the node exists in our connected component
        if node in G.nodes:
            hotspot_nodes.append(node)
            valid_centers.append(center)
            hotspot_info.append({
                'cluster_id': i,
                'node_id': node,
                'lat': center[0],
                'lon': center[1]
            })
            print(f"✅ Cluster {i}: Node {node} at ({center[0]:.4f}, {center[1]:.4f})")
        else:
            print(f"⚠️ Cluster {i}: Nearest node not in connected component")
    except Exception as e:
        print(f"⚠️ Cluster {i}: Error mapping to road network - {e}")

print(f"\nMapped {len(hotspot_nodes)} out of {len(centers)} hotspots to connected road network")

if len(hotspot_nodes) < 2:
    raise ValueError("Not enough connected hotspot nodes to build a route.")

# Test connectivity between all hotspot pairs
print("\nTesting connectivity between hotspot nodes...")
connectivity_matrix = {}
unreachable_pairs = []

for i, node1 in enumerate(hotspot_nodes):
    connectivity_matrix[node1] = {}
    for j, node2 in enumerate(hotspot_nodes):
        if i != j:
            try:
                distance = nx.shortest_path_length(G, node1, node2, weight='length')
                connectivity_matrix[node1][node2] = distance
            except nx.NetworkXNoPath:
                connectivity_matrix[node1][node2] = float('inf')
                unreachable_pairs.append((i, j, node1, node2))

if unreachable_pairs:
    print(f"⚠️ Found {len(unreachable_pairs)} unreachable pairs:")
    for i, j, node1, node2 in unreachable_pairs[:5]:  # Show first 5
        print(f"   Cluster {i} -> Cluster {j} (Node {node1} -> {node2})")
    
    # Find largest set of mutually connected nodes
    print("\nFinding largest set of mutually connected hotspots...")
    connected_hotspots = []
    
    # Start with first node and build connected set
    for start_node in hotspot_nodes:
        connected_set = [start_node]
        for candidate in hotspot_nodes:
            if candidate != start_node:
                # Check if candidate is reachable from all nodes in current set
                reachable = True
                for existing in connected_set:
                    if connectivity_matrix[existing][candidate] == float('inf'):
                        reachable = False
                        break
                    if connectivity_matrix[candidate][existing] == float('inf'):
                        reachable = False
                        break
                
                if reachable:
                    connected_set.append(candidate)
        
        if len(connected_set) > len(connected_hotspots):
            connected_hotspots = connected_set
    
    hotspot_nodes = connected_hotspots
    print(f"✅ Using {len(hotspot_nodes)} mutually connected hotspots for routing")

# Improved greedy routing with error handling
def safe_shortest_path_length(G, source, target):
    """Safely calculate shortest path length"""
    try:
        return nx.shortest_path_length(G, source, target, weight='length')
    except nx.NetworkXNoPath:
        return float('inf')

def safe_shortest_path(G, source, target):
    """Safely calculate shortest path"""
    try:
        return nx.shortest_path(G, source, target, weight='length')
    except nx.NetworkXNoPath:
        return []

print("\nCalculating optimal patrol route...")
visited = [hotspot_nodes[0]]
unvisited = set(hotspot_nodes[1:])
route_segments = []

while unvisited:
    last = visited[-1]
    
    # Find nearest unvisited node
    distances = {}
    for candidate in unvisited:
        dist = safe_shortest_path_length(G, last, candidate)
        if dist != float('inf'):
            distances[candidate] = dist
    
    if not distances:
        print("⚠️ No more reachable nodes. Ending route construction.")
        break
    
    # Select nearest reachable node
    next_node = min(distances.keys(), key=lambda x: distances[x])
    
    # Get path to next node
    segment = safe_shortest_path(G, last, next_node)
    
    if segment:
        route_segments.extend(segment[:-1])  # Avoid duplicating nodes
        visited.append(next_node)
        unvisited.remove(next_node)
        print(f"✅ Added segment to cluster at node {next_node} (distance: {distances[next_node]:.0f}m)")
    else:
        print(f"⚠️ Cannot create path to node {next_node}. Skipping.")
        unvisited.remove(next_node)

# Add final node
if visited:
    route_segments.append(visited[-1])

# Calculate total patrol distance
total_distance = 0
for i in range(len(route_segments) - 1):
    try:
        edge_data = G.get_edge_data(route_segments[i], route_segments[i + 1])
        if edge_data:
            # Handle multiple edges between nodes
            edge_length = min(data['length'] for data in edge_data.values()) if isinstance(edge_data, dict) else edge_data['length']
            total_distance += edge_length
    except:
        # If direct edge doesn't exist, calculate shortest path distance
        dist = safe_shortest_path_length(G, route_segments[i], route_segments[i + 1])
        if dist != float('inf'):
            total_distance += dist

print(f"\n✅ Patrol route completed!")
print(f"   Hotspots covered: {len(visited)}")
print(f"   Total route nodes: {len(route_segments)}")
print(f"   Total patrol distance: {total_distance:.2f} meters ({total_distance / 1000:.2f} km)")

# Create final patrol route map
print("\nGenerating patrol route visualization...")
m = folium.Map(location=[gdf['Latitude'].mean(), gdf['Longitude'].mean()], zoom_start=12)

# Add cluster centers as markers
for i, info in enumerate(hotspot_info):
    if info['node_id'] in visited:
        color = 'green'
        popup_text = f"Hotspot {info['cluster_id']} (Included)"
    else:
        color = 'red'
        popup_text = f"Hotspot {info['cluster_id']} (Excluded)"
    
    folium.Marker(
        location=[info['lat'], info['lon']], 
        icon=folium.Icon(color=color),
        popup=popup_text
    ).add_to(m)

# Add patrol route
if len(route_segments) > 1:
    route_coords = []
    for node in route_segments:
        if node in G.nodes:
            route_coords.append((G.nodes[node]['y'], G.nodes[node]['x']))
    
    if len(route_coords) > 1:
        folium.PolyLine(
            locations=route_coords, 
            color='red', 
            weight=4,
            popup=f"Patrol Route ({total_distance/1000:.2f} km)"
        ).add_to(m)

m.save("vancouver_patrol_route.html")
print("✅ Map saved as 'vancouver_patrol_route.html'")

# Summary statistics
print(f"\n📊 PATROL ROUTE SUMMARY:")
print(f"   Total crime hotspots identified: {len(centers)}")
print(f"   Hotspots accessible by road: {len(hotspot_nodes)}")
print(f"   Hotspots included in route: {len(visited)}")
print(f"   Route efficiency: {len(visited)/len(centers)*100:.1f}%")
print(f"   Total patrol distance: {total_distance/1000:.2f} km")
print(f"   Average distance per hotspot: {total_distance/len(visited)/1000:.2f} km")