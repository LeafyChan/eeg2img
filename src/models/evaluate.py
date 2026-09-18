"""
Validate SleepNetEncoder against NeuralBench's interface requirements
before running the full benchmark.
"""
from neuralbench import check_model, evaluate_model
from sleepnet_model import SleepNetEncoder

model = SleepNetEncoder()

# check_model pushes synthetic batches through — catches shape bugs in seconds
print("=== Shape check ===")
result = check_model(model, "eeg", "sleep_onset")
print(result)
