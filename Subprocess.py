##This is a batch automation script that runs Python files in the `scripts` folder sequentially without supervision.
import subprocess

# List of scripts to be executed in sequence
scripts = [
    r"The storage path for the 1.py file.",
    r"The storage path for the 2.py file.",
    r"The storage path for the 3.py file.",
    r'The storage path for the 4.py file.',
    r'The storage path for the 5.py file.',
    r'The storage path for the 6.py file.',
    r'The storage path for the 7.py file.'
]

# Iterate through each script and execute it
for script in scripts:
    try:
        # Use subprocess.run() to execute each script and wait for completion
        print(f"Executing {script} ...")
        result = subprocess.run(["python", script], check=True, text=True, capture_output=True)

        # If the script executes successfully
        print(f"{script} executed successfully.")
        print("Output:\n", result.stdout)  # Optional: Print the output of the script

    except subprocess.CalledProcessError as e:
        # If the script execution fails, print the error and stop further execution
        print(f"Error executing {script}: {e}")
        print("Error Output:\n", e.stderr)  # Optional: Print the error output
        break  # Stop execution of subsequent scripts if one fails
