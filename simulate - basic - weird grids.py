import os
import sys
from pathlib import Path
from io import StringIO

def run_simulations(times, boards, players, output_file):
    # Write a header for the results file
    with open(output_file, "w") as f:
        f.write("Time | Board | First Player | Second Player | Final Score\n")
        f.write("-" * 60 + "\n")

    for time_limit in times:
        for board in boards:
            for first, second in players:
                # Initialize counters for wins
                player1_wins = 0
                player2_wins = 0

                for i in range(20):
                    print(f"Running simulation {i+1} for time={time_limit}, board={Path(board).stem}, first={first}, second={second}...")

                    # Simulate command-line arguments
                    sys.argv = [
                        "simulate_game.py",
                        f"--first={first}",
                        f"--second={second}",
                        f"--board={board}",
                        f"--time={time_limit}"
                    ]

                    # Capture the output of simulate_game.main()
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()  # Redirect stdout to capture printed output

                    try:
                        simulate_game.main()  # Run the game simulation
                        output = sys.stdout.getvalue()  # Get the printed output
                    except Exception as e:
                        print(f"Simulation {i+1} failed: {str(e)}")
                        output = ""
                    finally:
                        sys.stdout = old_stdout  # Restore original stdout

                    # Check the output for win messages
                    if "Player 1 wins the game" in output:
                        player1_wins += 1
                        print("Player 1 wins")
                    elif "Player 2 wins the game" in output:
                        player2_wins += 1
                        print("Player 2 wins")

                # Log the final scores for this configuration
                with open(output_file, "a") as f:
                    f.write(f"{time_limit} | {Path(board).stem} | {first} | {second} | {player1_wins}-{player2_wins}\n")

if __name__ == '__main__':
    # Set the working directory to the project root
    os.chdir(r"C:\Users\Yosar\Desktop\2AMU10_Assigment_team05-main\competitive_sudoku")

    # Add the directory containing simulate_game.py to the Python path
    simulate_game_path = Path(r"C:\Users\Yosar\Desktop\2AMU10_Assigment_team05-main\competitive_sudoku")
    sys.path.append(str(simulate_game_path))

    # Import simulate_game module
    import simulate_game

    # Define parameters
    times = [0.1, 0.5, 1.0, 2.0, 5.0]
    boards = [
        r"C:/Users/Yosar/Desktop/2AMU10_Assigment_team05-main/competitive_sudoku/boards/empty-2x2.txt",
        r"C:/Users/Yosar/Desktop/2AMU10_Assigment_team05-main/competitive_sudoku/boards/empty-2x3.txt",
        r"C:/Users/Yosar/Desktop/2AMU10_Assigment_team05-main/competitive_sudoku/boards/empty-3x4.txt",
        r"C:/Users/Yosar/Desktop/2AMU10_Assigment_team05-main/competitive_sudoku/boards/empty-4x4.txt",
        r"C:/Users/Yosar/Desktop/2AMU10_Assigment_team05-main/competitive_sudoku/boards/empty-5x5.txt",
        r"C:/Users/Yosar/Desktop/2AMU10_Assigment_team05-main/competitive_sudoku/boards/empty-6x6.txt"

    ]
    players = [("team05_A1_Copy", "greedy_player"), ("tean05_A1_Copy", "team05_A2")]
    output_file = "simulation_results/simulation_summary_basic_vs_greedy_weird.txt"

    # Create the output folder if it doesn't exist
    os.makedirs("simulation_results", exist_ok=True)

    # Run simulations
    run_simulations(times, boards, players, output_file)
