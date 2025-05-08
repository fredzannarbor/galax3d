import numpy as np
import sys
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to read data from a text file
def read_data_from_file(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
        # Safely evaluate the string representation of the list
        data = ast.literal_eval(content)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Please ensure it exists in the current directory.")
        sys.exit(1)
    except SyntaxError as e:
        print(f"Error: Unable to parse the content of '{filename}'. Ensure it is formatted correctly.")
        print(f"Details: {e}")
        sys.exit(1)

# Function to filter data by maximum distance
def filter_by_distance(data, max_distance=None):
    if max_distance is None:
        return data
    return [row for row in data if row[2] <= max_distance]

# Get input file from command-line argument or use default
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = 'exoplanet_hosts.txt'  # Default file

# Read data from the file
data = read_data_from_file(input_file)

# Filter data by maximum distance (adjust as needed, set to None to disable)
max_distance = 10000  # Parsecs, or set to None to include all stars
filtered_data = filter_by_distance(data, max_distance)
#filtered_data = filtered_data[:1000]
if not filtered_data:
    print(f"Warning: No stars within {max_distance} parsecs. Using full dataset.")
    filtered_data = data
else:
    print(f"Filtered to {len(filtered_data)} stars within {max_distance} parsecs.")

# Extract columns from filtered data
l = np.array([row[0] for row in filtered_data])  # Galactic longitude (degrees)
b = np.array([row[1] for row in filtered_data])  # Galactic latitude (degrees)
d = np.array([row[2] for row in filtered_data])  # Distance (parsecs)
names = [row[3] for row in filtered_data]        # Object names

# Convert to radians
l_rad = np.radians(l)
b_rad = np.radians(b)

# Convert to Cartesian coordinates
x = d * np.cos(b_rad) * np.cos(l_rad)
y = d * np.cos(b_rad) * np.sin(l_rad)
z = d * np.sin(b_rad)

# Apply a scaling factor to increase spacing among stars
scaling_factor = 2.0  # Adjust this value to increase or decrease spacing
x = x * scaling_factor
y = y * scaling_factor
z = z * scaling_factor

# Assign colors based on the input file (to distinguish datasets)
if 'exoplanet' in input_file.lower():
    color = 'blue'
    dataset_label = 'Exoplanet Hosts'
elif 'local_bubble' in input_file.lower():
    color = 'green'
    dataset_label = 'Local Bubble Stars'
elif 'spinward' in input_file.lower():
    color = 'purple'
    dataset_label = 'Spinward Volume Stars'
else:
    color = 'cyan'
    dataset_label = 'Stars'

# Create hover text for all points
hover_text = [f"Name: {name}<br>Distance: {dist:.2f} pc<br>l: {lon:.2f}°<br>b: {lat:.2f}°"
              for name, dist, lon, lat in zip(names, d, l, b)]

# Create the 3D scatter plot with Plotly
fig = go.Figure()

# Plot stars
fig.add_trace(go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=3,  # Smaller size for large datasets
        color=color,
        opacity=0.6
    ),
    text=hover_text,
    hoverinfo='text',
    name=dataset_label
))

# Plot Earth (Sun) at origin
fig.add_trace(go.Scatter3d(
    x=[0],
    y=[0],
    z=[0],
    mode='markers',
    marker=dict(
        size=10,
        color='red',
        opacity=1.0
    ),
    text=["Earth (Sun)"],
    hoverinfo='text',
    name='Earth (Sun)'
))

# Limit annotations to avoid clutter (adjust as needed)
max_annotations = 10  # Maximum number of stars to annotate
for i in range(min(max_annotations, len(filtered_data))):
    fig.add_trace(go.Scatter3d(
        x=[x[i]],
        y=[y[i]],
        z=[z[i]],
        mode='text',
        text=[names[i]],
        textposition='top center',
        showlegend=False,
        hoverinfo='skip'
    ))

# Update layout for better visualization
fig.update_layout(
    title=f'3D Map of {dataset_label} (Source: {input_file})',
    scene=dict(
        xaxis_title='X (parsecs)',
        yaxis_title='Y (parsecs)',
        zaxis_title='Z (parsecs)',
        aspectmode='cube',  # Equal aspect ratio
        xaxis=dict(range=[-max(np.abs(x)) * 1.2, max(np.abs(x)) * 1.2]),
        yaxis=dict(range=[-max(np.abs(y)) * 1.2, max(np.abs(y)) * 1.2]),
        zaxis=dict(range=[-max(np.abs(z)) * 1.2, max(np.abs(z)) * 1.2])
    ),
    showlegend=True,
    width=1000,
    height=800
)

# Show the interactive plot
fig.show()