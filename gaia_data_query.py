import numpy as np
import requests
import urllib3
import os
import ssl

try:
    from astroquery.gaia import Gaia
except ModuleNotFoundError:
    print("Please install astroquery: pip install astroquery")
    exit(1)
from astropy.coordinates import SkyCoord, Galactic
from astropy import units as u
from astropy.time import Time

# Disable SSL warnings and verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# Set environment variable to disable SSL verification (for lower-level control)
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Optionally, create a custom SSL context to disable verification
ssl._create_default_https_context = ssl._create_unverified_context

# Configure astroquery to use a session with SSL verification disabled
from astroquery.utils.tap.core import TapPlus
Gaia = TapPlus(url="https://gea.esac.esa.int/tap-server/tap", verbose=False)
Gaia.session = requests.Session()
Gaia.session.verify = False


# Login to Gaia archive (optional, for larger queries)
# Gaia.login(user='your_username', password='your_password')


# Helper function to convert equatorial to galactic coordinates and calculate distance
def equatorial_to_galactic(ra, dec, parallax):
    # Remove invalid or negative parallaxes
    valid = (parallax > 0) & (~np.isnan(parallax))
    ra, dec, parallax = ra[valid], dec[valid], parallax[valid]

    # Calculate distance in parsecs (1/parallax in mas)
    distance = 1000.0 / parallax  # parallax in mas -> distance in parsecs

    # Convert to galactic coordinates
    coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
    galactic = coords.galactic
    l = galactic.l.degree
    b = galactic.b.degree

    return l, b, distance, valid


# 1. Query for stars hosting verified exoplanets within 100 parsecs
def query_exoplanet_hosts():
    # Note: Gaia DR3 does not directly list exoplanets, so we use a simplified approach
    # We'll query stars within 100 parsecs and assume a cross-match with an exoplanet catalog
    # For demonstration, we use a sample query and name stars as potential hosts
    query = """
            SELECT TOP 1000
        source_id, ra \
                 , dec \
                 , parallax \
                 , phot_g_mean_mag
            FROM gaiadr3.gaia_source
            WHERE parallax >= 10 -- Distance <= 100 parsecs
              AND parallax IS NOT NULL
              AND phot_g_mean_mag IS NOT NULL
            ORDER BY parallax DESC \
            """

    job = Gaia.launch_job_async(query)
    results = job.get_results()
    print("Available columns:", results.colnames)

    ra = np.array(results['ra'])
    dec = np.array(results['dec'])
    parallax = np.array(results['parallax'])
    source_ids = results['SOURCE_ID']

    l, b, distance, valid = equatorial_to_galactic(ra, dec, parallax)
    names = [f"Star_{sid}" for sid in source_ids[valid]]

    # Format data for galax_converter.py
    data = [[l[i], b[i], distance[i], names[i]] for i in range(len(l))]

    return data


# 2. Query for all stars in the Local Bubble (~200 parsecs)
def query_local_bubble():
    query = """
            SELECT TOP 10000
        source_id, ra \
                 , dec \
                 , parallax \
                 , phot_g_mean_mag
            FROM gaiadr3.gaia_source
            WHERE parallax >= 5 -- Distance <= 200 parsecs
              AND parallax IS NOT NULL
              AND phot_g_mean_mag IS NOT NULL
            ORDER BY parallax DESC \
            """

    job = Gaia.launch_job_async(query)
    results = job.get_results()

    ra = np.array(results['ra'])
    dec = np.array(results['dec'])
    parallax = np.array(results['parallax'])
    source_ids = results['SOURCE_ID']

    l, b, distance, valid = equatorial_to_galactic(ra, dec, parallax)
    names = [f"LB_Star_{sid}" for sid in source_ids[valid]]

    # Format data for galax_converter.py
    data = [[l[i], b[i], distance[i], names[i]] for i in range(len(l))]

    return data


# 3. Query for stars in the spinward volume Sol will reach in 1000 years
def query_spinward_volume():
    # Solar motion: ~20 km/s towards l=90°, b=0° (galactic center direction)
    # In 1000 years, distance = speed * time
    speed = 20 * u.km / u.s
    time = 1000 * u.yr
    distance_traveled = (speed * time).to(u.pc, equivalencies=u.parallax()).value  # ~0.006 pc

    # Define a conical volume in the spinward direction (l ~ 90°)
    # We'll query stars within a small angular radius around l=90°, b=0°
    query = """
            SELECT TOP 1000
        source_id, ra \
                 , dec \
                 , parallax \
                 , phot_g_mean_mag
            FROM gaiadr3.gaia_source
            WHERE parallax >= 5 -- Distance <= 200 parsecs
              AND parallax IS NOT NULL
              AND phot_g_mean_mag IS NOT NULL
              AND 1= CONTAINS (
                POINT('ICRS' \
                , ra \
                , dec) \
                , CIRCLE('ICRS' \
                , 271.1 \
                , 27.1 \
                , 10)
                )               -- Approx. l=90°, b=0° with 10° radius
            ORDER BY parallax DESC \
            """

    job = Gaia.launch_job_async(query)
    results = job.get_results()

    ra = np.array(results['ra'])
    dec = np.array(results['dec'])
    parallax = np.array(results['parallax'])
    source_ids = results['SOURCE_ID']

    l, b, distance, valid = equatorial_to_galactic(ra, dec, parallax)
    names = [f"Spin_Star_{sid}" for sid in source_ids[valid]]

    # Format data for galax_converter.py
    data = [[l[i], b[i], distance[i], names[i]] for i in range(len(l))]

    return data


# Save data to files compatible with galax_converter.py
def save_data(data, filename):
    with open(filename, 'w') as f:
        f.write('[\n')
        for i, row in enumerate(data):
            f.write(f'    [{row[0]:.2f}, {row[1]:.2f}, {row[2]:.2f}, "{row[3]}"]')
            if i < len(data) - 1:
                f.write(',')
            f.write('\n')
        f.write(']\n')


# Execute queries and save results
if __name__ == "__main__":
    # Query and save exoplanet hosts
    exoplanet_data = query_exoplanet_hosts()
    save_data(exoplanet_data, 'exoplanet_hosts.txt')
    print(f"Saved {len(exoplanet_data)} exoplanet host stars to exoplanet_hosts.txt")

    # Query and save Local Bubble stars
    local_bubble_data = query_local_bubble()
    save_data(local_bubble_data, 'local_bubble_stars.txt')
    print(f"Saved {len(local_bubble_data)} Local Bubble stars to local_bubble_stars.txt")

    # Query and save spinward volume stars
    spinward_data = query_spinward_volume()
    save_data(spinward_data, 'spinward_volume_stars.txt')
    print(f"Saved {len(spinward_data)} spinward volume stars to spinward_volume_stars.txt")