import wandb
import random
import time

print("--- Starting W&B Test Script ---")

try:
    # 1. Initialize a new W&B run.
    #    - project: A project is a collection of runs.
    #    - entity: Your W&B username or team name.
    #    - job_type: A way to group runs (e.g., "test", "train").
    run = wandb.init(
        project="Thesis-Iven",  # <--- IMPORTANT: Change this to your project name!
        entity="gorillawatch",  # <--- IMPORTANT: Change this to your W&B username!
        name="wandb_test_run",  # Optional: Name your run
    )
    print(f"--- W&B Run Initialized Successfully! ---")
    print(f"Run Name: {run.name}")
    print(f"Run URL: {run.url}")

    # 2. Log a simple configuration.
    wandb.config.update({
        "learning_rate": 0.01,
        "architecture": "TestNet",
        "slurm_job_id": run.settings.run_id # W&B automatically picks up SLURM_JOB_ID
    })
    print("--- Logged config ---")

    # 3. Log some dummy metrics in a loop.
    for i in range(10):
        wandb.log({"accuracy": 100 - i*2 - random.random(), "loss": i/10 + random.random()})
        print(f"Logged step {i}")
        time.sleep(0.5)

    # 4. Finish the run. This is crucial to mark the run as "finished".
    run.finish()
    print("--- W&B Run Finished Successfully! ---")

except Exception as e:
    print(f"!!! An error occurred during the W&B test: {e} !!!")
    # Exit with a non-zero status code to make Slurm aware of the failure
    exit(1)