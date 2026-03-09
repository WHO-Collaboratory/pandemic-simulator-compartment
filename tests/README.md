# Unit Tests for Pandemic Simulator Compartment Models

This directory contains comprehensive unit and integration tests for all disease models in the pandemic simulator.

## Test Structure

```
tests/
├── __init__.py           # Test package initialization
├── conftest.py           # Shared fixtures and configuration
├── test_models.py        # Tests for all disease models
├── test_validation.py    # Tests for validation schemas
├── test_helpers.py       # Tests for helper functions
└── test_integration.py   # End-to-end integration tests
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_models.py
```

### Run specific test class
```bash
pytest tests/test_models.py::TestABCModel
```

### Run specific test function
```bash
pytest tests/test_models.py::TestABCModel::test_model_initialization
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage report
```bash
pytest --cov=compartment --cov-report=html
```

### Run only fast tests (exclude slow ones)
```bash
pytest -m "not slow"
```

### Run only integration tests
```bash
pytest -m integration
```

## Test Categories

### Unit Tests (`test_models.py`)
- **TestABCModel**: Tests for ABC compartmental model
- **TestCovidModel**: Tests for COVID-19 SEIR model  
- **TestMpoxModel**: Tests for Mpox SEIR model
- **TestDengueModel**: Tests for Dengue vector-borne model
- **TestModelComparison**: Cross-model comparison tests

### Validation Tests (`test_validation.py`)
- **TestABCValidation**: Pydantic schema validation for ABC
- **TestCovidValidation**: Validation for COVID config
- **TestMpoxValidation**: Validation for Mpox config
- **TestDengueValidation**: Validation for Dengue config
- **TestValidationEdgeCases**: Edge cases and error handling

### Helper Tests (`test_helpers.py`)
- **TestLogging**: Logging setup tests
- **TestInterventions**: Intervention function tests
- **TestNumericHelpers**: JAX/NumPy operations
- **TestDateHelpers**: Date manipulation utilities

### Integration Tests (`test_integration.py`)
- **TestABCIntegration**: Full ABC simulation pipeline
- **TestCovidIntegration**: COVID end-to-end tests
- **TestMpoxIntegration**: Mpox integration tests
- **TestDengueIntegration**: Dengue integration tests
- **TestCrossModelIntegration**: Multi-model consistency

## Fixtures

Common fixtures defined in `conftest.py`:

- `base_config`: Base configuration dictionary
- `abc_config`: ABC model configuration
- `covid_config`: COVID model configuration
- `mpox_config`: Mpox model configuration
- `dengue_config`: Dengue model configuration
- `load_example_config`: Loads example config files

## Writing New Tests

### Example unit test:
```python
def test_my_feature(abc_config):
    """Test description"""
    model = ABCJaxModel(abc_config)
    
    # Test something
    assert model.alpha > 0
```

### Example integration test:
```python
@pytest.mark.integration
def test_full_simulation(abc_config):
    """Test complete simulation"""
    model = ABCJaxModel(abc_config)
    result = model.simulate()
    
    assert result is not None
    assert "compartment_data" in result
```

## Best Practices

1. **Use descriptive test names**: Test names should clearly describe what is being tested
2. **Use fixtures**: Leverage fixtures for common setup code
3. **Test edge cases**: Include tests for boundary conditions and error cases
4. **Keep tests isolated**: Each test should be independent
5. **Use appropriate assertions**: Choose the right assertion for the check
6. **Add docstrings**: Document what each test verifies

## Dependencies

Required packages for testing:
- `pytest >= 6.0`
- `pytest-cov` (optional, for coverage reports)
- All packages in `pyproject.toml`

Install test dependencies:
```bash
pip install pytest pytest-cov
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.13'
      - run: pip install -e .
      - run: pip install pytest pytest-cov
      - run: pytest --cov=compartment
```

## Troubleshooting

### Import errors
Make sure the package is installed in development mode:
```bash
pip install -e .
```

### JAX errors
Ensure JAX is properly installed for your platform:
```bash
pip install jax jaxlib
```

### Missing fixtures
Check that `conftest.py` is in the tests directory and properly formatted.
