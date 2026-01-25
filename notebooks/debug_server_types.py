
import pandas as pd
import os
from pathlib import Path

# Path to data
data_path = Path(r"c:\pyproj\AWS-CapacityForecaster\data\scratch\server_metrics.csv.gz")

if not data_path.exists():
    print(f"File not found: {data_path}")
    # Try looking in local relative path if project root assumed differently
    data_path = Path("../data/scratch/server_metrics.csv.gz")

if data_path.exists():
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path, compression='gzip')
    
    if 'server_type' in df.columns:
        server_types = df['server_type'].unique()
        print(f"Server Types found: {server_types}")
        
        # Check case insensitive
        web_mask = df['server_type'].astype(str).str.lower() == 'web'
        web_servers = df[web_mask]['server_id'].unique()
        print(f"Web servers found (case-insensitive): {len(web_servers)}")
        if len(web_servers) > 0:
            print(f"First web server: {web_servers[0]}")
        else:
            print("No web servers found.")
    else:
        print("Column 'server_type' not found.")
        print("Columns:", df.columns)
else:
    print("Data file could not be located.")
