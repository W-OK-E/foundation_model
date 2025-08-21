import subprocess
import sys
import requests

# -----------------------
# SLACK CONFIG
# -----------------------
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"  # <-- replace with yours

def notify(message):
    try:
        payload = {"text": f"<@U09B23R0CQ1> {message}"}  # replace with your Slack User ID
        requests.post(SLACK_WEBHOOK, json=payload)
    except Exception as e:
        print("Slack notification failed:", e)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 monitor.py '<command to run>'")
        sys.exit(1)

    command = sys.argv[1]
    print(f"[Monitor] Running: {command}")

    try:
        # Capture both stdout + stderr
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=True
        )
        notify(f"✅ SUCCESS on *{subprocess.getoutput('hostname')}*:\n```\n{result.stdout[-500:]}\n```")
    except subprocess.CalledProcessError as e:
        error_tail = (e.stderr or e.stdout)[-500:]  # last 500 chars of error
        notify(
            f"❌ FAILED on *{subprocess.getoutput('hostname')}* "
            f"(exit {e.returncode}):\n```\n{error_tail}\n```"
        )

if __name__ == "__main__":
    main()
