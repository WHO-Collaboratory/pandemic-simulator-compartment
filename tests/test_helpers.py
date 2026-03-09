"""Unit tests for helper functions"""
import pytest
import numpy as np
import jax.numpy as jnp
from datetime import datetime

from compartment.helpers import setup_logging
from compartment.interventions import jax_timestep_intervention, jax_prop_intervention


class TestLogging:
    """Tests for logging setup"""
    
    def test_setup_logging_runs(self):
        """Test that logging setup doesn't raise errors"""
        # Should not raise any exceptions
        setup_logging()


class TestInterventions:
    """Tests for intervention functions"""
    
    def test_timestep_intervention_no_intervention(self):
        """Test timestep intervention with no active interventions"""
        interventions = []
        current_ordinal = datetime(2024, 1, 15).toordinal()
        rates = {"beta": 0.5}
        original_rates = {"beta": 0.5}
        
        updated_rates = jax_timestep_intervention(
            interventions, 
            current_ordinal, 
            rates, 
            original_rates
        )
        
        assert updated_rates["beta"] == 0.5
    
    def test_timestep_intervention_active_intervention(self):
        """Test timestep intervention with active intervention"""
        start_date = datetime(2024, 1, 10)
        current_date = datetime(2024, 1, 15)
        
        interventions = [{
            "start_date": start_date,
            "end_date": datetime(2024, 1, 20),
            "intervention_type": "transmission_reduction",
            "value": 0.5,  # 50% reduction
            "admin_zones": "all"
        }]
        
        current_ordinal = current_date.toordinal()
        rates = {"beta": 0.5}
        original_rates = {"beta": 0.5}
        
        updated_rates = jax_timestep_intervention(
            interventions,
            current_ordinal,
            rates,
            original_rates
        )
        
        # Rate should be reduced
        assert updated_rates["beta"] < original_rates["beta"]
    
    def test_prop_intervention_no_intervention(self):
        """Test proportional intervention with no active interventions"""
        interventions = []
        current_ordinal = datetime(2024, 1, 15).toordinal()
        rates = {"beta": 0.5}
        original_rates = {"beta": 0.5}
        
        updated_rates = jax_prop_intervention(
            interventions,
            current_ordinal,
            rates,
            original_rates
        )
        
        assert updated_rates["beta"] == 0.5
    
    def test_intervention_restoration_after_end_date(self):
        """Test that rates are restored after intervention ends"""
        start_date = datetime(2024, 1, 10)
        end_date = datetime(2024, 1, 15)
        current_date = datetime(2024, 1, 20)  # After intervention
        
        interventions = [{
            "start_date": start_date,
            "end_date": end_date,
            "intervention_type": "transmission_reduction",
            "value": 0.5,
            "admin_zones": "all"
        }]
        
        current_ordinal = current_date.toordinal()
        rates = {"beta": 0.3}  # Currently reduced
        original_rates = {"beta": 0.5}
        
        updated_rates = jax_timestep_intervention(
            interventions,
            current_ordinal,
            rates,
            original_rates
        )
        
        # Rate should be restored to original
        assert updated_rates["beta"] == original_rates["beta"]


class TestNumericHelpers:
    """Tests for numeric helper functions"""
    
    def test_jax_array_operations(self):
        """Test basic JAX array operations"""
        arr = jnp.array([1.0, 2.0, 3.0])
        
        assert jnp.sum(arr) == 6.0
        assert jnp.mean(arr) == 2.0
    
    def test_numpy_jax_compatibility(self):
        """Test conversion between NumPy and JAX arrays"""
        np_array = np.array([1.0, 2.0, 3.0])
        jax_array = jnp.array(np_array)
        
        assert jnp.allclose(jax_array, np_array)
    
    def test_matrix_multiplication(self):
        """Test matrix operations"""
        a = jnp.array([[1, 2], [3, 4]])
        b = jnp.array([[5, 6], [7, 8]])
        
        result = jnp.matmul(a, b)
        
        expected = jnp.array([[19, 22], [43, 50]])
        assert jnp.allclose(result, expected)


class TestDateHelpers:
    """Tests for date manipulation"""
    
    def test_datetime_to_ordinal(self):
        """Test datetime to ordinal conversion"""
        date = datetime(2024, 1, 1)
        ordinal = date.toordinal()
        
        assert isinstance(ordinal, int)
        assert ordinal > 0
    
    def test_ordinal_to_datetime(self):
        """Test ordinal to datetime conversion"""
        date = datetime(2024, 1, 1)
        ordinal = date.toordinal()
        reconstructed = datetime.fromordinal(ordinal)
        
        assert reconstructed.year == date.year
        assert reconstructed.month == date.month
        assert reconstructed.day == date.day
    
    def test_date_arithmetic(self):
        """Test date arithmetic"""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        
        diff = (end - start).days
        
        assert diff == 30
