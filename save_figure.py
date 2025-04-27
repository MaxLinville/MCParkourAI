import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def save_figure():
    # Read CSV file, skipping parameter rows
    df = pd.read_csv('fitness_metrics_5_block.csv', skiprows=9)
    # Create figure with appropriate size
    plt.figure(figsize=(12, 6))

    # Plot best fitness
    plt.plot(df['Generation'], df['Best Fitness'], 'b-', marker='o', linewidth=2, label='Best Fitness')

    # Plot average fitness
    plt.plot(df['Generation'], df['Average Fitness'], 'r-', marker='x', linewidth=2, label='Average Fitness')

    # Add trendlines
    # z1 = np.polyfit(df['Generation'], df['Best Fitness'], 1)
    # p1 = np.poly1d(z1)
    # plt.plot(df['Generation'], p1(df['Generation']), "b--", alpha=0.7, label='Best Fitness Trend')

    # z2 = np.polyfit(df['Generation'], df['Average Fitness'], 1)
    # p2 = np.poly1d(z2)
    # plt.plot(df['Generation'], p2(df['Generation']), "r--", alpha=0.7, label='Avg Fitness Trend')

    # Add labels and title
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.title('Genetic Algorithm Performance Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    # Annotate highest fitness point
    best_gen = df['Best Fitness'].idxmax()
    plt.annotate(f'Max: {df["Best Fitness"].max()}', 
                xy=(df.loc[best_gen, 'Generation'], df.loc[best_gen, 'Best Fitness']),
                xytext=(5, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

    # Save plot
    plt.tight_layout()
    plt.savefig('fitness_over_time_5_block.png')
    plt.show()

    print("Plot created and saved as 'fitness_over_time.png'")