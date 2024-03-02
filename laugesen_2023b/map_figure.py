import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, box

dark_blue = '#0072B2'

# Locations with corrected 'Goulburn Weir' coordinates and 'Eildon Dam'
locations = {
    'Goulburn Weir': (145.17, -36.717),  # Corrected coordinates
    'Taggerty': (145.75, -37.25),
    'Shepparton': (145.39, -36.38),
    'Echuca': (144.75, -36.13),
    'Eildon Dam': (145.896944, -37.192778)
}

# Convert locations to GeoDataFrame
gdf = gpd.GeoDataFrame({
    'Location': locations.keys(),
    'Coordinates': [Point(xy) for xy in locations.values()]
}, geometry='Coordinates')

# Apply a 20km buffer (in degrees) to the bounding box
buffer_in_degrees = 20 / 111  # Approximate conversion
bounds = gdf.total_bounds  # minx, miny, maxx, maxy
expanded_bounds = box(bounds[0] - buffer_in_degrees, bounds[1] - buffer_in_degrees,
                      bounds[2] + buffer_in_degrees, bounds[3] + buffer_in_degrees)

# Load the world map and filter for Australia
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
australia_only = world[world['name'] == 'Australia']

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))
world.plot(ax=ax, color='white', edgecolor=dark_blue)
gdf.plot(ax=ax, color=dark_blue, markersize=50)
ax.grid(True, which='both', color='lightgrey', linestyle='--', linewidth=0.5)

# Labeling the points
for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.Location):
    ax.text(x, y, '  ' + label, fontsize=14, ha='left', va='center', color=dark_blue)

# Inset Map showing only Australia
inset_ax = fig.add_axes([0.65, 0.65, 0.3, 0.3])
australia_only.plot(ax=inset_ax, color='white', edgecolor=dark_blue, linewidth=2)
gpd.GeoSeries([expanded_bounds]).boundary.plot(ax=inset_ax, color='black', linewidth=1)
inset_ax.set_xlim(110, 155)
inset_ax.set_ylim(-45, -10)
inset_ax.axis('off')

ax.set_xlim(bounds[0] - buffer_in_degrees, bounds[2] + buffer_in_degrees)
ax.set_ylim(bounds[1] - buffer_in_degrees, bounds[3] + buffer_in_degrees)
#ax.set_title('Detailed Locations in Victoria, Australia with Labels', fontsize=15)
plt.axis('off')

#plt.savefig('/path/to/save/australia_map_labeled_corrected_final_with_buffer.png', format='png', bbox_inches='tight')
