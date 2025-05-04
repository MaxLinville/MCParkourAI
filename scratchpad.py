import os
import pandas as pd
import glob
from datetime import timedelta

def count_total_generations(metrics_folder="fitness_metrics"):
    """
    Count the total number of generations across all CSV files in the metrics folder.
    
    Args:
        metrics_folder (str): Path to the folder containing CSV files
        
    Returns:
        dict: Dictionary with 'total_generations' and breakdown by file
    """
    # Check if the folder exists
    if not os.path.exists(metrics_folder):
        print(f"Error: Folder '{metrics_folder}' not found")
        return {"total_generations": 0, "files": {}}
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(metrics_folder, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {metrics_folder}")
        return {"total_generations": 0, "files": {}}
    
    total_generations = 0
    file_generations = {}
    
    print(f"Found {len(csv_files)} CSV files:")
    
    # Process each file
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        try:
            # Skip parameter rows (usually the first 9 rows in your format)
            df = pd.read_csv(csv_file, skiprows=9)
            
            # Count generations (rows) in this file
            num_generations = len(df)
            
            # Check if we have actual data
            if num_generations > 0:
                file_generations[filename] = num_generations
                total_generations += num_generations
                print(f"  - {filename}: {num_generations} generations")
            else:
                print(f"  - {filename}: No generation data found")
                
        except Exception as e:
            print(f"  - Error reading {filename}: {str(e)}")
    
    return {"total_generations": total_generations, "files": file_generations}

def calculate_training_time(total_generations, agents_per_generation=48, time_per_agent=30):
    """
    Calculate the total training time based on generations, agents, and time per agent.
    
    Args:
        total_generations (int): Total number of generations
        agents_per_generation (int): Number of agents evaluated per generation
        time_per_agent (float): Average time in seconds to evaluate one agent
        
    Returns:
        dict: Dictionary with training time in different formats
    """
    # Calculate total seconds
    total_seconds = total_generations * agents_per_generation * time_per_agent
    
    # Convert to timedelta for easier formatting
    total_time = timedelta(seconds=total_seconds)
    
    # Format as days, hours, minutes, seconds
    days = total_time.days
    hours, remainder = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Create formatted strings
    time_str = f"{days} days, {hours:02d}:{minutes:02d}:{seconds:02d}"
    hours_total = days * 24 + hours + minutes/60 + seconds/3600
    
    return {
        "total_seconds": total_seconds,
        "time_str": time_str,
        "days": days,
        "hours": hours,
        "minutes": minutes,
        "seconds": seconds,
        "total_hours": hours_total
    }

def analyze_training_time_for_file(file_path, agents_per_generation=48, time_per_agent=30):
    """
    Calculate the training time for a single CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        agents_per_generation (int): Number of agents evaluated per generation
        time_per_agent (float): Average time in seconds to evaluate one agent
        
    Returns:
        dict: Dictionary with training information for this file
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        # Extract parameters from CSV headers
        with open(file_path, 'r') as f:
            header_lines = [next(f) for _ in range(10)]
        
        parameters = {}
        for line in header_lines:
            if line.startswith('# '):
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    key = parts[0].replace('# ', '')
                    value = parts[1]
                    parameters[key] = value
        
        # Read generation data
        df = pd.read_csv(file_path, skiprows=9)
        num_generations = len(df)
        
        # Calculate training time
        agents = int(parameters.get('Population Size', agents_per_generation))
        time_info = calculate_training_time(num_generations, agents, time_per_agent)
        
        # Add file-specific information
        latest_gen = df['Generation'].max() if not df.empty else 0
        best_fitness = df['Best Fitness'].max() if not df.empty else 0
        best_gen = df.loc[df['Best Fitness'].idxmax()]['Generation'] if not df.empty else 0
        
        return {
            "parameters": parameters,
            "generations": num_generations,
            "latest_generation": latest_gen,
            "best_fitness": best_fitness,
            "best_fitness_generation": best_gen,
            "training_time": time_info
        }
        
    except Exception as e:
        return {"error": str(e)}

def main():
    # Get total generations across all files
    results = count_total_generations()
    total_gens = results["total_generations"]
    
    print(f"\nTotal generations across all files: {total_gens}")
    
    # Calculate the total training time with default parameters
    training_time = calculate_training_time(total_gens)
    
    print(f"\nEstimated total training time:")
    print(f"- {training_time['time_str']} (≈ {training_time['total_hours']:.2f} hours)")
    print(f"- Total evaluations: {total_gens * 48} agent runs")
    
    # Example of analyzing a specific file
    specific_file = "fitness_metrics/fitness_metrics_new_params_old_map.csv"
    if os.path.exists(specific_file):
        print(f"\nAnalyzing specific file: {os.path.basename(specific_file)}")
        file_analysis = analyze_training_time_for_file(specific_file)
        
        if "error" not in file_analysis:
            time_info = file_analysis["training_time"]
            print(f"- Generations: {file_analysis['generations']}")
            print(f"- Latest generation: {file_analysis['latest_generation']}")
            print(f"- Best fitness: {file_analysis['best_fitness']} (at generation {file_analysis['best_fitness_generation']})")
            print(f"- Training time: {time_info['time_str']} (≈ {time_info['total_hours']:.2f} hours)")
            print(f"- Parameters:")
            for key, value in file_analysis["parameters"].items():
                print(f"  • {key}: {value}")
        else:
            print(f"  Error: {file_analysis['error']}")
    
    # Prompt to analyze a different file
    print("\nWould you like to analyze a specific CSV file? (y/n)")
    choice = input().lower()
    
    if choice == 'y':
        print("Enter the filename (in the fitness_metrics folder, e.g. fitness_metrics_new_params_old_map.csv):")
        filename = input().strip()
        if not filename.startswith("fitness_metrics/"):
            filename = f"fitness_metrics/{filename}"
        
        print("\nEnter agents per generation (default: 48):")
        try:
            agents_input = input().strip()
            agents = int(agents_input) if agents_input else 48
        except:
            agents = 48
            
        print("Enter average time per agent in seconds (default: 30):")
        try:
            time_input = input().strip()
            time_per_agent = float(time_input) if time_input else 30
        except:
            time_per_agent = 30
            
        file_analysis = analyze_training_time_for_file(filename, agents, time_per_agent)
        
        if "error" not in file_analysis:
            time_info = file_analysis["training_time"]
            print(f"\nResults for {os.path.basename(filename)}:")
            print(f"- Generations: {file_analysis['generations']}")
            print(f"- Training time: {time_info['time_str']} (≈ {time_info['total_hours']:.2f} hours)")
        else:
            print(f"\nError: {file_analysis['error']}")

if __name__ == "__main__":
    main()