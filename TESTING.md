# Testing Guide

This document provides a comprehensive guide to testing the Pandemic Simulator Compartment Models.

## Quick Start

### 1. Install Test Dependencies

```bash
# Install with test dependencies
pip install -e ".[test]"

# Or install pytest manually
pip install pytest pytest-cov
```

### 2. Run Tests

```bash
# Run all tests
pytest

# Or use the convenience script
./run_tests.sh
```

## Test Organization

The test suite is organized into four main categories:

### 1. **Model Tests** (`test_models.py`)
Tests for individual disease models and their implementations:
- Model initialization and configuration
- Population matrix dimensions
- Transmission rate validation
- Simulation execution
- Cross-model comparisons

**Example:**
```python
def test_model_initialization(abc_config):
    """Test that ABC model initializes correctly"""
    model = ABCJaxModel(abc_config)
    assert model.alpha == 0.3
    assert model.beta == 0.1
```

### 2. **Validation Tests** (`test_validation.py`)
Tests for Pydantic validation schemas:
- Config structure validation
- Required field enforcement
- Type checking
- Edge case handling

**Example:**
```python
def test_negative_transmission_rate_fails():
    """Test that negative transmission rates are rejected"""
    edge_data = {
        "source": "a_compartment",
        "target": "b_compartment",
        "data": {"transmission_rate": -0.3}
    }
    with pytest.raises(ValidationError):
        ABCTransmissionEdge(**edge_data)
```

### 3. **Helper Tests** (`test_helpers.py`)
Tests for utility functions:
- Logging setup
- Intervention functions
- Numeric operations
- Date handling

**Example:**
```python
def test_timestep_intervention_active_intervention():
    """Test timestep intervention with active intervention"""
    interventions = [{
        "start_date": datetime(2024, 1, 10),
        "end_date": datetime(2024, 1, 20),
        "intervention_type": "transmission_reduction",
        "value": 0.5,
        "admin_zones": "all"
    }]
    # Test intervention logic...
```

### 4. **Integration Tests** (`test_integration.py`)
End-to-end workflow tests:
- Config loading and processing
- Full simulation pipelines
- Output serialization
- Cross-model consistency

**Example:**
```python
def test_full_simulation_pipeline(load_example_config, tmp_path):
    """Test complete simulation from config to output"""
    config = load_example_config("abc_jax_model")
    processed_config = process_config(config)
    result = run_simulation(processed_config)
    assert result is not None
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run specific test class
pytest tests/test_models.py::TestABCModel

# Run specific test
pytest tests/test_models.py::TestABCModel::test_model_initialization
```

### Using the Run Script

```bash
# Run all tests
./run_tests.sh

# Run with verbose output
./run_tests.sh --verbose

# Run fast tests only (skip slow ones)
./run_tests.sh --fast

# Generate coverage report
./run_tests.sh --coverage

# Run specific pattern
./run_tests.sh --pattern ABC
./run_tests.sh --pattern test_validation

# Combine options
./run_tests.sh --verbose --coverage --pattern COVID
```

### Advanced Options

```bash
# Run with detailed output
pytest -vv

# Show print statements
pytest -s

# Stop at first failure
pytest -x

# Run last failed tests
pytest --lf

# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Generate XML report for CI
pytest --junitxml=test-results.xml
```

## Coverage Reports

### Generate HTML Coverage Report

```bash
pytest --cov=compartment --cov-report=html

# View report
open htmlcov/index.html
```

### Terminal Coverage Report

```bash
pytest --cov=compartment --cov-report=term
```

### Coverage Targets

Aim for:
- **Overall**: >80% coverage
- **Models**: >90% coverage
- **Validation**: >95% coverage
- **Helpers**: >85% coverage

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

| Fixture | Description |
|---------|-------------|
| `base_config` | Base configuration dictionary |
| `abc_config` | ABC model configuration |
| `covid_config` | COVID model configuration |
| `mpox_config` | Mpox model configuration |
| `dengue_config` | Dengue model configuration |
| `load_example_config` | Function to load example configs |
| `example_config_path` | Path getter for example configs |

**Usage:**
```python
def test_my_feature(abc_config):
    model = ABCJaxModel(abc_config)
    # ... test code
```

## Writing New Tests

### Test Structure

```python
import pytest
from compartment.models.abc_jax_model.model import ABCJaxModel

class TestMyFeature:
    """Tests for my feature"""
    
    def test_something(self, abc_config):
        """Test description goes here"""
        # Arrange
        model = ABCJaxModel(abc_config)
        
        # Act
        result = model.some_method()
        
        # Assert
        assert result == expected_value
```

### Best Practices

1. **Use descriptive names**: `test_transmission_rate_cannot_be_negative`
2. **One assertion per test**: Focus on testing one thing
3. **Use fixtures**: Avoid duplicating setup code
4. **Test edge cases**: Include boundary conditions
5. **Document tests**: Add clear docstrings
6. **Keep tests fast**: Use small timesteps for simulations
7. **Make tests independent**: Don't rely on test execution order

### Common Assertions

```python
# Equality
assert result == expected

# Approximate equality (for floats)
assert result == pytest.approx(expected, rel=1e-5)
assert jnp.isclose(result, expected)

# Boolean conditions
assert condition is True
assert value > 0

# Exceptions
with pytest.raises(ValueError):
    dangerous_function()

# Container membership
assert item in collection
assert len(collection) == 5

# Type checking
assert isinstance(obj, MyClass)
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.13']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
    
    - name: Run tests with coverage
      run: |
        pytest --cov=compartment --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Install package in development mode
pip install -e .
```

**JAX/NumPy issues:**
```bash
# Reinstall JAX
pip install --upgrade jax jaxlib
```

**Fixture not found:**
```bash
# Ensure conftest.py exists in tests directory
ls tests/conftest.py
```

**Slow tests:**
```bash
# Reduce timesteps in test configs
abc_config["time_steps"] = 10  # Instead of 100
```

### Debug Mode

```bash
# Run with Python debugger
pytest --pdb

# Drop into debugger on failure
pytest -x --pdb

# More verbose error output
pytest --tb=long
```

## Performance Testing

### Timing Tests

```bash
# Show slowest tests
pytest --durations=10

# Profile tests
pytest --profile
```

### Memory Testing

```bash
# Install memory profiler
pip install pytest-memprof

# Run with memory profiling
pytest --memprof
```

## Test Markers

Use markers to organize tests:

```python
@pytest.mark.slow
def test_long_simulation():
    """This test takes a while"""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Integration test"""
    pass
```

Run marked tests:
```bash
pytest -m slow
pytest -m "not slow"
pytest -m integration
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)

## Getting Help

- Check test output for error messages
- Review test documentation in docstrings
- Run with `-vv` for detailed output
- Use `--pdb` to debug failing tests
