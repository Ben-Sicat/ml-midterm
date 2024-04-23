import subprocess

# Install dependencies from requirements.txt
subprocess.run(["pip", "install", "-r", "requirments.txt"])

# Train the model (assuming train.py is the script)
subprocess.run(["python", "train.py"])

# Predict sentiment (assuming predict.py is the script)
subprocess.run(["python", "predict.py"])

print("All steps completed!")
