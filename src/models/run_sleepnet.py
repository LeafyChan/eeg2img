import sys
sys.path.insert(0, '/home/leafy/NeuralBench/src/models')

from sleepnet_model import SleepNetEncoder
from neuralbench import check_model, evaluate_model

model = SleepNetEncoder()

# Print layer names so we know what to pass
print("=== Layer names ===")
for name, _ in model.named_children():
    print(f"  {name}")

print("\n=== Shape check ===")
print(check_model(model, "eeg", "sleep_onset"))

print("\n=== Running evaluation (all layers unfrozen) ===")
scores = evaluate_model(
    model,
    "eeg",
    "sleep_onset",
    name="sleepnet-v1-fulltune",
    dataset="alvarez2022haaglanden",
    overrides={"downstream_model_wrapper.layers_to_unfreeze": ["projections", "temporal"]}
)
print(scores)
