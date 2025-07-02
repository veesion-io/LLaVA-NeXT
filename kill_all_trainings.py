#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import subprocess
import json
import logging

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from veesion_data_core.tools import stop_training, destroy_training

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command):
    """Runs a command and returns its stdout, handling errors."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command: {' '.join(command)}")
        logging.error(f"Stderr: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        logging.error(f"Command not found: {command[0]}. Is it installed and in your PATH?")
        return None

def get_active_training_runs():
    """Fetches the display titles of active training runs from GitHub Actions."""
    logging.info("Finding active training runs from GitHub Actions...")
    statuses = ["in_progress", "queued"]
    all_runs = []

    for status in statuses:
        command = [
            "gh", "run", "list",
            "--repo", "veesion-io/terraform-scalable-training",
            "--workflow", "Run scalable training",
            "--status", status,
            "--json", "displayTitle"
        ]
        output = run_command(command)
        if not output:
            logging.warning(f"Could not get active workflow runs with status '{status}' from GitHub.")
            continue

        try:
            runs = json.loads(output)
            all_runs.extend(runs)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON output for status '{status}' from the gh command.")

    if not all_runs:
        logging.info("No active or queued training runs found.")
        return []

    training_names = [run['displayTitle'] for run in all_runs]
    if training_names:
        logging.info(f"Found active or queued training runs: {', '.join(training_names)}")
    else:
        logging.info("No active or queued training runs found.")
    return training_names

def stop_training_run(training_name):
    """Stops a training run using a Python script."""
    logging.info(f"Attempting to stop training: {training_name}...")
    command = [
        "python3",
        "-c",
        f"from veesion_data_core.tools import stop_training; stop_training('{training_name}')"
    ]
    output = run_command(command)
    if output is not None:
        logging.info(f"Successfully sent stop command for {training_name}.")
    else:
        logging.error(f"Failed to send stop command for {training_name}.")

def main():
    """Main function to find and kill all active trainings."""
    training_names = get_active_training_runs()
    if not training_names:
        return

    for name in training_names:
        stop_training_run(name)

if __name__ == "__main__":
    main() 