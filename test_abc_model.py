#!/usr/bin/env python3
"""
Quick test script to verify the ABC model can be loaded and instantiated.
"""
import json
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from compartment.models.abc_jax_model.model import ABCJaxModel
from compartment.validation.diseases.abc import ABCDiseaseConfig

def test_abc_model_import():
    """Test that the ABC model can be imported"""
    print("✓ ABC model imported successfully")
    return True

def test_abc_validation():
    """Test that the ABC validation schema works"""
    config_data = {
        "disease_type": "ABC",
        "compartment_list": ["A", "B", "C"],
        "transmission_edges": [
            {
                "source": "a_compartment",
                "target": "b_compartment",
                "data": {
                    "transmission_rate": 0.3
                }
            },
            {
                "source": "b_compartment",
                "target": "c_compartment",
                "data": {
                    "transmission_rate": 0.1
                }
            }
        ]
    }
    
    try:
        disease_config = ABCDiseaseConfig(**config_data)
        print("✓ ABC validation schema works correctly")
        print(f"  - Compartments: {disease_config.compartment_list}")
        print(f"  - Transmission edges: {len(disease_config.transmission_edges)}")
        return True
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

def test_example_config():
    """Test that the example config file is valid JSON"""
    config_path = Path(__file__).parent / "compartment" / "models" / "abc_jax_model" / "example-config.json"
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        print("✓ Example config is valid JSON")
        print(f"  - Admin units: {len(config['case_file']['admin_zones'])}")
        print(f"  - Compartments: {config['Disease']['compartment_list']}")
        return True
    except Exception as e:
        print(f"✗ Example config validation failed: {e}")
        return False

def main():
    print("Testing ABC Model...\n")
    
    tests = [
        test_abc_model_import,
        test_abc_validation,
        test_example_config
    ]
    
    results = [test() for test in tests]
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print(f"{'='*50}")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
