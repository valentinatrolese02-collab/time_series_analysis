import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Reverted to your specific pipeline path
output_dir = Path("ex2/exercise2_3_outputs")

def perform_task_2_4_analysis():
    # 1. Load the matrices and states
    try:
        # Fixed the TypeError: changed 'index_index' back to 'index_col'[cite: 2]
        A = pd.read_csv(output_dir / "exercise2_3_A_matrix.csv", index_col=0).values
        B = pd.read_csv(output_dir / "exercise2_3_B_matrix.csv", index_col=0).values
        states = pd.read_csv(output_dir / "exercise2_3_filtered_states.csv")
        
        # Kept the original filename as per your initial script
        data = pd.read_csv("transformer_data.csv")
    except FileNotFoundError as e:
        print(f"Error: Missing files. Ensure {output_dir} and transformer_data.csv exist. {e}")
        return

    print("--- Matrix Analysis ---")
    # 2. Steady State Gain Analysis (I - A)^-1 * B
    I_mat = np.eye(2)
    steady_state_gain = np.linalg.inv(I_mat - A) @ B
    
    gain_df = pd.DataFrame(
        steady_state_gain, 
        index=["State 1 (Temp)", "State 2 (Hidden)"], 
        columns=["Gain_Ta", "Gain_S", "Gain_I"]
    )
    print("\nSteady-State Gain Matrix:")
    print(gain_df)

    # 3. Correlation Analysis[cite: 1]
    analysis_df = pd.concat([states, data[["Ta", "S", "I"]]], axis=1)
    correlations = analysis_df.corr()[["state_1", "state_2"]]
    
    print("\nCorrelations between States and Inputs:")
    print(correlations)

    # 4. Visualization for Task 2.4: States + Inputs[cite: 1]
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    
    # State Trajectories
    axes[0].plot(states['time'], states['state_1'], label='State 1 (Temp)', color='tab:blue')
    axes[0].plot(states['time'], states['state_2'], label='State 2 (Buffer)', color='tab:orange', linestyle='--')
    axes[0].set_ylabel("States")
    axes[0].set_title("Reconstructed States and Input Variables")
    axes[0].legend(loc='upper right')

    # Input Variables[cite: 1]
    axes[1].plot(states['time'], data['Ta'], color='tab:green')
    axes[1].set_ylabel("Ta (°C)")
    
    axes[2].plot(states['time'], data['S'], color='tab:red')
    axes[2].set_ylabel("Solar (W/m²)")
    
    axes[3].plot(states['time'], data['I'], color='tab:purple')
    axes[3].set_ylabel("Load (kA)")
    axes[3].set_xlabel("Time [hours]")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__":
    perform_task_2_4_analysis()