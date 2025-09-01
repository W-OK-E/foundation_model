import subprocess
import sys
import requests

# -----------------------
# SLACK CONFIG
# -----------------------
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"  # <-- replace with yours

SLACK_USER_ID = "<@U09B23R0CQ1>"  # replace with your Slack ID if needed

def notify(message):
    if not SLACK_WEBHOOK:
        print("Missing SLACK_WEBHOOK env variable.")
        return
    try:
        payload = {"text": f"{SLACK_USER_ID} {message}"}
        response = requests.post(SLACK_WEBHOOK, json=payload)
        response.raise_for_status()
    except Exception as e:
        print("Slack notification failed:", e)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 monitor.py '<command to run>'")
        sys.exit(1)

    # Join all command-line arguments into a single command string
    command = " ".join(sys.argv[1:])
    print(f"[Monitor] Running: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        hostname = subprocess.getoutput("hostname")

        if result.returncode == 0:
            notify(
                f"✅ SUCCESS on *{hostname}*:\nCommand: `{command}`"
            )
        else:
            notify(
                f"❌ FAILED on *{hostname}*:\nCommand: `{command}`\nError: ```{result.stderr.strip()}```"
            )

    except Exception as e:
        notify(f"❌ Exception during execution:\n```{str(e)}```")

if __name__ == "__main__":
    main()
