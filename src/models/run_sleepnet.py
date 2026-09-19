import sys
sys.path.insert(0, '/home/leafy/NeuralBench/src/models')

from sleepnet_model import SleepNetEncoder
from neuralbench import check_model, evaluate_model

model = SleepNetEncoder()

print("=== Shape check ===")
print(check_model(model, "eeg", "sleep_onset"))

print("\n=== Running evaluation ===")
scores = evaluate_model(
    model,
    "eeg",
    "sleep_onset",
    name="sleepnet-v1",
    dataset="alvarez2022haaglanden",
    debug=False
)
print(scores)
