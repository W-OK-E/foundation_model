import subprocess
import pkg_resources

output = []

for dist in pkg_resources.working_set:
    name = dist.project_name
    try:
        # Use pip show to get version
        result = subprocess.run(["pip", "show", name], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                version = line.split(":", 1)[1].strip()
                output.append(f"{name}=={version}")
                break
    except Exception as e:
        print(f"Error processing {name}: {e}")

# Write to file
with open("requirements_clean.txt", "w") as f:
    f.write("\n".join(sorted(output)))

print("✅ Clean requirements written to requirements_clean.txt")
