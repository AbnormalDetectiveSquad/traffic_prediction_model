import os
import subprocess

def sync_wandb_runs(wandb_dir):
    """
    Synchronize all offline-run directories in the given wandb directory.

    :param wandb_dir: Path to the wandb directory
    """
    if not os.path.exists(wandb_dir):
        print(f"Error: Directory {wandb_dir} does not exist.")
        return

    for root, dirs, files in os.walk(wandb_dir):
        for dir_name in dirs:
            if dir_name.startswith("offline-run"):
                run_path = os.path.join(root, dir_name)
                print(f"Syncing {run_path}...")
                try:
                    result = subprocess.run(["wandb", "sync", run_path], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"Successfully synced: {run_path}")
                    else:
                        print(f"Error syncing {run_path}: {result.stderr}")
                except FileNotFoundError:
                    print("Error: WandB CLI not found. Ensure WandB is installed and accessible.")
                    return

def main():
    wandb_dir = "./wandb"  # Change this to your wandb directory path if needed
    sync_wandb_runs(wandb_dir)

if __name__ == "__main__":
    main()
