#!/usr/bin/env python3
"""Debug script to check ABC model validation"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from compartment.validation import SimulationConfig, ABCDiseaseConfig
from pydantic import ValidationError

# Load the example config
config_path = Path(__file__).parent / "compartment" / "models" / "abc_jax_model" / "example-config.json"

with open(config_path) as f:
    raw_config = json.load(f)

# Wrap it like the actual code does
wrapped_config = {
    "data": {
        "getSimulationJob": raw_config
    }
}

print("=" * 60)
print("Testing ABC Model Validation")
print("=" * 60)

try:
    # Try to validate using the same approach as the code
    validated_config = SimulationConfig[ABCDiseaseConfig](**wrapped_config["data"]["getSimulationJob"])
    print("✓ Validation succeeded!")
    print(f"  Compartments: {validated_config.Disease.compartment_list}")
    print(f"  Disease Type: {validated_config.Disease.disease_type}")
except ValidationError as e:
    print("✗ Validation failed:")
    print()
    for error in e.errors():
        loc = " -> ".join(str(x) for x in error['loc'])
        print(f"  Field: {loc}")
        print(f"  Error: {error['msg']}")
        print(f"  Type: {error['type']}")
        print()
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
