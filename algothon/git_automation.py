import subprocess
import time

REPO_PATH = r"C:\Users\rhirl\man-imperial-algothon-2026"
BRANCH = "main"
CHECK_INTERVAL = 10  # seconds

def get_remote_commit():
    result = subprocess.run(
        ["git", "ls-remote", "origin", BRANCH],
        cwd=REPO_PATH,
        capture_output=True,
        text=True
    )
    return result.stdout.split()[0]

def get_local_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_PATH,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def pull_repo():
    subprocess.run(["git", "pull", "--ff-only"], cwd=REPO_PATH)

print("Watching repo for updates...")

while True:
    try:
        remote_commit = get_remote_commit()
        local_commit = get_local_commit()

        if remote_commit != local_commit:
            print("New commit detected. Pulling...")
            pull_repo()
        else:
            print("No changes.")

    except Exception as e:
        print("Error:", e)

    time.sleep(CHECK_INTERVAL)