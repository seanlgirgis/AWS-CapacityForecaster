
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# -------------------------------------------------------------------------
# 1. SETUP ENVIRONMENT
# -------------------------------------------------------------------------
# Add the project 'src' directory to Python path so we can import our modules
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    # print(f"Added {src_path} to sys.path")

try:
    from utils.server_archetypes import ServerArchetype, ServerType, ARCHETYPE_PROFILES
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configure logging to avoid "No handler found" warnings
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------------
# 2. CONFIGURATION
# -------------------------------------------------------------------------
SERVER_ID = "server_playground_01"
ARCHETYPE = "web"  # Try: 'web', 'database', 'application', 'batch'
DAYS_TO_SIMULATE = 7

# -------------------------------------------------------------------------
# 3. DEEP DIVE: UNDERSTANDING THE RANDOMNESS
# -------------------------------------------------------------------------
def explain_randomness_process(server_type_str, server_id):
    """
    Walks through the math of how randomness and correlations are generated.
    """
    print(f"\n{'='*60}")
    print(f"[DEEP DIVE]: How {server_type_str.upper()} Server Randomness Works")
    print(f"{'='*60}")
    
    # 1. Get the profile
    server_type_enum = {
        'web': ServerType.WEB, 
        'database': ServerType.DATABASE,
        'application': ServerType.APPLICATION,
        'batch': ServerType.BATCH
    }[server_type_str]
    
    profile = ARCHETYPE_PROFILES[server_type_enum]
    
    print(f"1. PROFILE ATTRIBUTES (The 'DNA' of the server):")
    print(f"    - Base CPU:       {profile.base_cpu}%  (Variance: {profile.cpu_variance})")
    print(f"    - Base Network:   {profile.base_network} Mbps (Variance: {profile.network_variance})")
    print(f"    - CPU-Network Corr: {profile.cpu_network_correlation} (High correlation expected for Web)")
    
    # 2. The Math: Correlation Matrix
    print(f"\n2. THE MATH: Correlation Matrix Construction")
    print("    We define a matrix that describes how metrics move together.")
    # Reconstructing the matrix as it's done inside the class
    corr_matrix = np.array([
        [1.0, profile.cpu_memory_correlation, 0.1, profile.cpu_network_correlation],
        [profile.cpu_memory_correlation, 1.0, profile.memory_disk_correlation, 0.2],
        [0.1, profile.memory_disk_correlation, 1.0, 0.3],
        [profile.cpu_network_correlation, 0.2, 0.3, 1.0]
    ])
    # CPU, MEM, DISK, NET
    print(f"    Correlation Matrix (4x4 - CPU, Mem, Disk, Net):\n{np.round(corr_matrix, 2)}")

    # 3. Cholesky Decomposition
    print(f"\n3. THE ENGINE: Cholesky Decomposition (L @ z)")
    try:
        L = np.linalg.cholesky(corr_matrix)
        print("    We decompose this matrix into a Lower triangular matrix 'L'.")
        print(f"    L Matrix:\n{np.round(L, 2)}")
    except np.linalg.LinAlgError:
        print("    Matrix not positive definite, using Identity.")
        L = np.eye(4)
        
    # 4. Generating Randomness
    # Replicating the step for demonstration
    rng = np.random.RandomState(42)  # Fixed seed for demonstration
    z = rng.randn(4)  # 4 independent random numbers (Gaussian)
    
    print(f"\n4. THE STEP: Combining Base + Noise")
    print(f"    a. We roll 4 pure independent random numbers (z): {np.round(z, 2)}")
    
    # Transform
    correlated_noise = L @ z
    print(f"    b. We mix them using L (L @ z) to get Correlated Noise: {np.round(correlated_noise, 2)}")
    
    # Apply to CPU (Index 0)
    # CPU = Base + (CorrelatedNoise[0] * Variance)
    cpu_noise_val = correlated_noise[0]
    final_cpu_noise = cpu_noise_val * profile.cpu_variance
    print(f"    c. CPU Calculation (Example):")
    print(f"       Base: {profile.base_cpu}")
    print(f"       + Noise ({cpu_noise_val:.2f} * {profile.cpu_variance}): {final_cpu_noise:.2f}")
    result = profile.base_cpu + final_cpu_noise
    print(f"       = Result: {result:.2f} (Before clipping/time factors)")
    
    print(f"\n    (This happens for every single timestamp!)")

# -------------------------------------------------------------------------
# 4. SIMULATION & VISUALIZATION
# -------------------------------------------------------------------------
def run_simulation(server_type, server_id, days):
    print(f"\n{'='*60}")
    print(f"[START] RUNNING SIMULATION: {days} Days for {server_id}")
    print(f"{'='*60}")
    
    # Initialize Server
    archetype = ServerArchetype(
        {
            'web': ServerType.WEB, 
            'database': ServerType.DATABASE,
            'application': ServerType.APPLICATION,
            'batch': ServerType.BATCH
        }[server_type], 
        server_id
    )
    
    # Generate Timestamps (Hourly)
    start_date = pd.Timestamp("2024-01-01")
    timestamps = pd.date_range(start=start_date, periods=days*24, freq='h')
    
    data = []
    
    print(f"Generating {len(timestamps)} data points...")
    for ts in timestamps:
        # Get Time Factor (Business Hours / Weekend)
        time_factor = archetype.get_time_factor(ts)
        
        # Generate Metrics
        metrics = archetype.generate_correlated_metrics(
            timestamp=ts,
            time_factor=time_factor,
            trend_factor=0.0 # Ignore trend for short sim
        )
        
        metrics['timestamp'] = ts
        metrics['time_factor'] = time_factor # Save this to visualize it too
        data.append(metrics)
        
    df = pd.DataFrame(data)
    df = df.set_index('timestamp')
    
    print(f"[COMPLETE] Generation Complete. Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df[['cpu_p95', 'net_in_p95', 'time_factor']].head())
    
    # Verify Correlation
    print(f"\n[VERIFICATION]: Actual Correlations in Generated Data")
    corr_actual = df[['cpu_p95', 'mem_p95', 'disk_p95', 'net_in_p95']].corr()
    print(corr_actual.round(2))
    print(f"\nNote: CPU ({server_type}) should be correlated with Network as per profile.")
    
    # Plot
    plot_simulation(df, server_type)

def plot_simulation(df, server_type):
    # Set style
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(12, 12))
    
    # CPU
    plt.subplot(4, 1, 1)
    plt.plot(df.index, df['cpu_p95'], label='CPU %', color='royalblue')
    plt.title(f"{server_type.upper()} Server - CPU Utilization (Correlated with Network)", fontsize=14)
    plt.ylabel('Util %')
    plt.legend(loc='upper right')
    
    # Memory
    plt.subplot(4, 1, 2)
    plt.plot(df.index, df['mem_p95'], label='Memory %', color='orange')
    plt.title(f"{server_type.upper()} Server - Memory Utilization", fontsize=14)
    plt.ylabel('Util %')
    plt.legend(loc='upper right')

    # Disk
    plt.subplot(4, 1, 3)
    plt.plot(df.index, df['disk_p95'], label='Disk %', color='green')
    plt.title(f"{server_type.upper()} Server - Disk Utilization", fontsize=14)
    plt.ylabel('Util %')
    plt.legend(loc='upper right')

    # Network
    plt.subplot(4, 1, 4)
    plt.plot(df.index, df['net_in_p95'], label='Network In (Mbps)', color='purple')
    plt.title(f"{server_type.upper()} Server - Network Utilization (Correlated with CPU)", fontsize=14)
    plt.ylabel('Mbps')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save
    filename = "server_simulation_metrics.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"\n[SAVED] Plot saved to: {os.path.abspath(filename)}")
    print("Open this image to see the correlated patterns and business hour cycles!")

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    explain_randomness_process(ARCHETYPE, SERVER_ID)
    run_simulation(ARCHETYPE, SERVER_ID, DAYS_TO_SIMULATE)
