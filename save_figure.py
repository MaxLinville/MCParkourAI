import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def save_figure(file_name='updated_map'):
    # Check if file_name already includes the directory path
    if not file_name.startswith('fitness_metrics/'):
        csv_path = f'fitness_metrics/fitness_metrics_{file_name}.csv'
    else:
        csv_path = file_name
        if not csv_path.endswith('.csv'):
            csv_path = f'{csv_path}.csv'
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found at {csv_path}")
        return
    
    # Read CSV file, skipping parameter rows
    df = pd.read_csv(csv_path, skiprows=9)
    
    # Create figure with appropriate size
    plt.figure(figsize=(12, 6))

    # Plot best fitness
    plt.plot(df['Generation'], df['Best Fitness'], 'b-', marker='o', linewidth=2, label='Best Fitness')

    # Plot average fitness
    plt.plot(df['Generation'], df['Average Fitness'], 'r-', marker='x', linewidth=2, label='Average Fitness')

    # Add labels and title
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.title('Genetic Algorithm Performance Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    # Annotate highest fitness point
    best_gen = df['Best Fitness'].idxmax()
    plt.annotate(f'Max: {df["Best Fitness"].max():.2f}', 
                xy=(df.loc[best_gen, 'Generation'], df.loc[best_gen, 'Best Fitness']),
                xytext=(5, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

    # Create fitness_plots directory if it doesn't exist
    plots_dir = 'fitness_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract filename from the CSV path for use in the output filename
    input_filename = os.path.basename(csv_path)
    output_filename = f'fitness_plot_{file_name}.png'
    output_path = os.path.join(plots_dir, output_filename)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    print(f"Plot created and saved as '{output_path}'")

if __name__ == "__main__":
    save_figure()