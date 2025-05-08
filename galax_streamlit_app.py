import traceback

import streamlit as st
import numpy as np
import requests
import urllib3
import os
import ssl
import json
from astroquery.utils.tap.core import TapPlus
from astropy.coordinates import SkyCoord, Galactic
from astropy import units as u
import plotly.graph_objects as go
from dotenv import load_dotenv
from scipy.spatial import ConvexHull


# Disable SSL warnings and verification to handle certificate issues
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

# Configure astroquery to use a session with SSL verification disabled
Gaia = TapPlus(url="https://gea.esac.esa.int/tap-server/tap", verbose=True)
Gaia.session = requests.Session()
Gaia.session.verify = False

# Try to import OpenAI client; if not available, disable AI parsing
try:
    from openai import OpenAI

    openai_available = True
except ImportError:
    openai_available = False
    st.warning("OpenAI client not installed. AI parsing disabled. Install with `pip install openai` to enable.")

load_dotenv()
api_key = os.getenv("XAI_API_KEY")
base_url = os.getenv("BASE_URL")
#st.info(api_key)
#st.info(base_url)
st.session_state.query = None
# Helper function to convert equatorial to galactic coordinates and calculate distance
def equatorial_to_galactic(ra, dec, parallax):
    valid = (parallax > 0) & (~np.isnan(parallax))
    ra, dec, parallax = ra[valid], dec[valid], parallax[valid]
    distance = 1000.0 / parallax  # parallax in mas -> distance in parsecs
    coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
    galactic = coords.galactic
    l = galactic.l.degree
    b = galactic.b.degree
    return l, b, distance, valid


# Function to save data to a text file
def save_data(data, filename):
    with open(filename, 'w') as f:
        f.write('[\n')
        for i, row in enumerate(data):
            f.write(f'    [{row[0]:.2f}, {row[1]:.2f}, {row[2]:.2f}, "{row[3]}"]')
            if i < len(data) - 1:
                f.write(',')
            f.write('\n')
        f.write(']\n')
    return filename


# Function to use OpenAI API to parse free text into structured query parameters or ADQL
def parse_text_with_openai(description, api_key):
    if not openai_available:
        return None, "OpenAI-compatible client not available. Falling back to manual parsing."

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Simplify the prompt to reduce potential formatting issues
        prompt = """
Parse the following description into a structured query for Gaia DR3. Return a JSON object with:
- parameters: containing limit (int), min_parallax (float), region (string or null)
- adql_query: a complete ADQL query string
- explanation: brief explanation of interpretation

Description: "{description}"
"""
        prompt = prompt.format(description=description)

        response = client.chat.completions.create(
            model="grok-3-latest",
            messages=[
                {"role": "system", "content": "You are a precise astronomical data query assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500,
            response_format={ "type": "json_object" }  # Request JSON format explicitly
        )

        response_content = response.choices[0].message.content
        
        # Add debug logging
        st.write(f"LLM Response: {response_content}")
        
        try:
            parsed_response = json.loads(response_content)
            return parsed_response, "Successfully parsed with LLM."
        except json.JSONDecodeError as e:
            st.error(f"JSON Parse Error: {str(e)}")
            st.code(response_content, language="json")  # Display the malformed response
            return None, f"Failed to parse LLM response as JSON: {str(e)}"
            
    except Exception as e:
        st.error(f"Full error: {str(e)}")
        return None, f"LLM API error: {str(e)}"


def find_column(results, possible_names):
    for name in possible_names:
        if name in results.colnames:
            return name
    raise KeyError(f"None of {possible_names} found in results columns: {results.colnames}")

# Function to construct Gaia query (either from OpenAI or manual parsing)
def construct_gaia_query(description, api_key):
    # Try OpenAI parsing if API key is provided and client is available
    if api_key and openai_available:
        with st.spinner("Parsing description with LLM ..."):
            parsed_response, status = parse_text_with_openai(description, api_key)
            if parsed_response:
                params = parsed_response.get("parameters", {})
                adql_query = parsed_response.get("adql_query", "")
                st.info(f"LLM Interpretation: {parsed_response.get('explanation', 'No explanation provided.')}")
                if adql_query:
                    st.session_state.query = adql_query
                    return adql_query, params
                # If no full query, build one from parameters
                query = f"""
                    SELECT TOP {params.get('limit', 1000)}
                        SOURCE_ID, ra, dec, parallax, phot_g_mean_mag
                    FROM gaiadr3.gaia_source
                    WHERE parallax >= {params.get('min_parallax', 5.0)} -- Distance limit
                      AND parallax IS NOT NULL
                      AND phot_g_mean_mag IS NOT NULL
                """
                if params.get("ra_center") and params.get("dec_center"):
                    query += f"""
                      AND 1 = CONTAINS(
                        POINT('ICRS', ra, dec),
                        CIRCLE('ICRS', {params.get('ra_center')}, {params.get('dec_center')}, {params.get('radius', 10.0)})
                      )
                    """
                additional = params.get("additional_conditions", "")
                if additional:
                    query += f" AND {additional}"
                query += " ORDER BY parallax DESC"
                st.session_state.query = query
                return query, params
            else:
                st.warning(f"LL parsing failed: {status}. Falling back to manual parsing.")

    # Manual fallback parsing if LLM is not used or fails
    description = description.lower()
    query_params = {"limit": 1000, "min_parallax": 5, "region": None, "ra_center": None, "dec_center": None,
                    "radius": 10}

    if "within" in description and "parsecs" in description:
        try:
            dist_str = description.split("within")[1].split("parsecs")[0].strip()
            dist = float(dist_str)
            query_params["min_parallax"] = 1000.0 / dist
        except (IndexError, ValueError):
            st.warning("Could not parse distance. Defaulting to 200 parsecs.")
            query_params["min_parallax"] = 5

    if "local bubble" in description:
        query_params["limit"] = 5000
    elif "spinward" in description or "galactic center" in description:
        query_params["region"] = "spinward"
        query_params["ra_center"] = 271.1
        query_params["dec_center"] = 27.1
        query_params["radius"] = 10
    elif "exoplanet" in description or "planet" in description:
        query_params["region"] = "exoplanet"
        query_params["limit"] = 1000

    query = f"""
        SELECT TOP {query_params['limit']}
            SOURCE_ID, ra, dec, parallax, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE parallax >= {query_params['min_parallax']} -- Distance limit
          AND parallax IS NOT NULL
          AND phot_g_mean_mag IS NOT NULL
    """
    if query_params["region"] and query_params["ra_center"]:
        query += f"""
          AND 1 = CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {query_params['ra_center']}, {query_params['dec_center']}, {query_params['radius']})
          )
        """
    query += " ORDER BY parallax DESC"
    return query, query_params


def generate_figure_caption(query, data_stats):
    """Generate a scientific caption for the star visualization using LLM."""
    if not openai_available:
        return "3D visualization of stars from Gaia DR3 catalog"

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        prompt = f"""Generate a brief, one-sentence scientific caption for a 3D visualization of stars.
Query: {query}
Number of stars: {data_stats['total_stars']}
Distance range: {data_stats['min_distance']} to {data_stats['max_distance']} parsecs

The caption should be concise and technical, focusing on the key aspects of the visualization. Maximum 100 characters.
"""

        response = client.chat.completions.create(
            model="grok-3-latest",
            messages=[
                {"role": "system", "content": "You are a professional astronomer writing figure captions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )

        caption = response.choices[0].message.content.strip()
        return caption.rstrip('.')  # Remove trailing period if present

    except Exception as e:
        st.warning(f"Caption generation failed: {str(e)}")
        return "3D visualization of stars from Gaia DR3 catalog"


# Streamlit UI
st.title("Galax3D Star Query and Visualization")
st.markdown(
    "_Built with Grok, Streamlit and Astroquery for Gaia DR3 data retrieval. Optional LLM  integration using xAI API for query parsing._")

st.markdown("Enter a description of the stars you want to query from the Gaia archive at ESA, and visualize them in 3D.")

# Input field for L API key (optional)
#api_key = st.text_input("OpenAI API Key (optional, for AI parsing of description)", type="password", value="")
if api_key:
    st.info("Using LLM for query parsing. Key provided as XAI_API_KEY.")
else:
    st.info("No LLM API key provided. Using basic keyword-based parsing.")

# Input field for user description
user_input = st.text_area(
    "Describe the stars you want (e.g., 'stars within 100 parsecs with exoplanets', 'stars in the Local Bubble'):",
    value="stars within 100 parsecs")

# Button to trigger the query
if st.button("Fetch Stars"):
    with st.spinner("Constructing query and fetching data from Gaia..."):
        query, params = construct_gaia_query(user_input, api_key)
        st.code(query, language="sql")  # Display the generated query for transparency



        try:
            job = Gaia.launch_job_async(query)
            results = job.get_results()
            st.write(results)
            st.success(f"Query successful! Retrieved {len(results)} stars.")
            st.write("Available columns:", results.colnames)  # This will show all available column names
            # Process results
            ra = np.array(results['ra'])
            dec = np.array(results['dec'])
            parallax = np.array(results['parallax'])
            source_id_col = find_column(results, ['SOURCE_ID', 'source_id', 'Source_ID'])
            source_ids = results[source_id_col]


            l, b, distance, valid = equatorial_to_galactic(ra, dec, parallax)
            prefix = params.get("region", "Star").capitalize() if params.get("region") else "Star"
            names = [f"{prefix}_{sid}" for sid in source_ids[valid]]

            # Format data
            data = [[l[i], b[i], distance[i], names[i]] for i in range(len(l))]

            # Save to file
            output_file = "queried_stars.txt"
            save_data(data, output_file)
            st.success(f"Data saved to {output_file}")

            # Store data in session state for visualization
            st.session_state['star_data'] = data
            st.session_state['output_file'] = output_file

        except Exception as e:
            st.error(f"Error fetching data from Gaia: {str(e)}")
            # insert traceback
            st.write(traceback.format_exc())

# Visualization section (only shown if data is available)
if 'star_data' in st.session_state:
    st.header("Visualize Stars in 3D")
    scaling_factor = st.slider("Scaling Factor for Spacing", 1.0, 5.0, 2.0, 0.1)
    max_distance = st.number_input("Max Distance (parsecs)", min_value=10.0, max_value=500.0, value=50.0, step=5.0)
    max_annotations = st.number_input("Max Annotations", min_value=0, max_value=50, value=10, step=1)
    show_envelope = st.checkbox("Show Volume Envelope", value=False)

    # Filter data by max distance
    data = st.session_state['star_data']
    filtered_data = [row for row in data if row[2] <= max_distance]
    if not filtered_data:
        st.warning(f"No stars within {max_distance} parsecs. Showing all stars.")
        filtered_data = data
    else:
        st.info(f"Showing {len(filtered_data)} stars within {max_distance} parsecs.")

    data_stats = {
        "total_stars": len(filtered_data),
        "min_distance": f"{min(row[2] for row in filtered_data):.1f}",
        "max_distance": f"{max(row[2] for row in filtered_data):.1f}"
    }


    l = np.array([row[0] for row in filtered_data])
    b = np.array([row[1] for row in filtered_data])
    d = np.array([row[2] for row in filtered_data])
    names = [row[3] for row in filtered_data]

    # Convert to Cartesian coordinates
    l_rad = np.radians(l)
    b_rad = np.radians(b)
    x = d * np.cos(b_rad) * np.cos(l_rad)
    y = d * np.cos(b_rad) * np.sin(l_rad)
    z = d * np.sin(b_rad)

    # Apply scaling factor
    x = x * scaling_factor
    y = y * scaling_factor
    z = z * scaling_factor

    # Create hover text
    hover_text = [f"Name: {name}<br>Distance: {dist:.2f} pc<br>l: {lon:.2f}°<br>b: {lat:.2f}°"
                  for name, dist, lon, lat in zip(names, d, l, b)]

    # Create the 3D scatter plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.6),
        text=hover_text,
        hoverinfo='text',
        name='Stars'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='red', opacity=1.0),
        text=["Earth (Sun)"],
        hoverinfo='text',
        name='Earth (Sun)'
    ))

    # Add annotations for limited stars
    for i in range(min(max_annotations, len(filtered_data))):
        fig.add_trace(go.Scatter3d(
            x=[x[i]], y=[y[i]], z=[z[i]],
            mode='text',
            text=[names[i]],
            textposition='top center',
            showlegend=False,
            hoverinfo='skip'
        ))

    # Update layout
    max_coord = max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z))) if len(x) > 0 else 1
    # Generate the caption
    figure_caption = generate_figure_caption(st.session_state.query, data_stats)

    #figure_caption = "3-D map"
    # Use the caption in your figure layout
    fig.update_layout(title=figure_caption,
        scene=dict(
            xaxis_title='X (parsecs)',
            yaxis_title='Y (parsecs)',
            zaxis_title='Z (parsecs)',
            aspectmode='cube',
            xaxis=dict(range=[-max_coord * 1.2, max_coord * 1.2]),
            yaxis=dict(range=[-max_coord * 1.2, max_coord * 1.2]),
            zaxis=dict(range=[-max_coord * 1.2, max_coord * 1.2])
        ),
        showlegend=True,
        width=800,
        height=600
    )

    if show_envelope:
        envelope_color = st.color_picker("Envelope Color", "#0000FF")  # Default blue
        envelope_opacity = st.slider("Envelope Opacity", 0.0, 1.0, 0.2, 0.1)
        envelope_type = st.selectbox("Envelope Type", ["Convex Hull", "Alpha Shape"])

        try:
            points = np.column_stack((x, y, z))

            if envelope_type == "Convex Hull":
                hull = ConvexHull(points)
                vertices = points[hull.vertices]

                fig.add_trace(go.Mesh3d(
                    x=points[hull.vertices, 0],
                    y=points[hull.vertices, 1],
                    z=points[hull.vertices, 2],
                    opacity=envelope_opacity,
                    color=envelope_color,
                    name='Volume Envelope'
                ))
            else:  # Alpha Shape
                from alphashape import alphashape

                alpha = st.slider("Alpha Value", 0.0, 2.0, 1.0, 0.1)
                hull = alphashape(points, alpha)
                # Extract vertices and faces from the alpha shape
                # Note: Implementation depends on the returned type from alphashape
                # You might need to adjust this part based on the actual output

                fig.add_trace(go.Mesh3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    opacity=envelope_opacity,
                    color=envelope_color,
                    name='Alpha Shape'
                ))

        except Exception as e:
            st.warning(f"Could not create envelope: {str(e)}")

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display table of filtered data
    # Display table of filtered data
    st.header("Filtered Stars Data Table")
    # Convert filtered_data to a dictionary by including all columns
    table_data = {}
    for i, row in enumerate(filtered_data):
        # If this is the first row, initialize the dictionary with column names
        if i == 0:
            for j in range(len(row)):
                table_data[f"Column_{j}"] = []
        # Add each value to its respective column
        for j, value in enumerate(row):
            table_data[f"Column_{j}"].append(value)

    st.dataframe(table_data, use_container_width=True, hide_index=True)

st.markdown("---")

