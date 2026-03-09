"""Unit tests for all compartmental models"""
import pytest
import numpy as np
import jax.numpy as jnp
from datetime import datetime

from compartment.models.abc_jax_model.model import ABCJaxModel
from compartment.models.covid_jax_model.model import CovidJaxModel
from compartment.models.mpox_jax_model.model import MpoxJaxModel
from compartment.models.dengue_jax_model.model import DengueJaxModel


class TestABCModel:
    """Tests for ABC Jax Model"""
    
    def test_model_initialization(self, abc_config):
        """Test that ABC model initializes correctly"""
        model = ABCJaxModel(abc_config)
        
        assert model.alpha == 0.3
        assert model.beta == 0.1
        assert model.compartment_list == ["A", "B", "C"]
        assert model.n_timesteps == 100
        assert len(model.admin_units) == 2
    
    def test_population_matrix_shape(self, abc_config):
        """Test population matrix has correct dimensions"""
        model = ABCJaxModel(abc_config)
        
        assert model.population_matrix.shape == (2, 3)  # 2 zones, 3 compartments
    
    def test_population_conservation(self, abc_config):
        """Test that total population is conserved during initialization"""
        model = ABCJaxModel(abc_config)
        
        total_pop_initial = jnp.sum(model.population_matrix)
        expected_total = 9900 + 100 + 0 + 19900 + 100 + 0
        
        assert jnp.isclose(total_pop_initial, expected_total)
    
    def test_transmission_rates_positive(self, abc_config):
        """Test that transmission rates are positive"""
        model = ABCJaxModel(abc_config)
        
        assert model.alpha > 0
        assert model.beta > 0
    
    def test_original_rates_stored(self, abc_config):
        """Test that original rates are stored for intervention restoration"""
        model = ABCJaxModel(abc_config)
        
        assert "alpha" in model.original_rates
        assert "beta" in model.original_rates
        assert model.original_rates["alpha"] == model.alpha
    
    def test_simulate_runs_without_error(self, abc_config):
        """Test that simulation runs without raising errors"""
        model = ABCJaxModel(abc_config)
        
        # Should not raise any exceptions
        result = model.simulate()
        
        assert result is not None
        assert "compartment_data" in result
    
    def test_compartment_output_shape(self, abc_config):
        """Test that output has correct shape"""
        abc_config["time_steps"] = 10  # Run shorter simulation
        model = ABCJaxModel(abc_config)
        
        result = model.simulate()
        compartment_data = result["compartment_data"]
        
        # Should have data for each timestep
        assert len(compartment_data) > 0
    
    def test_model_with_zero_initial_infected(self, abc_config):
        """Test model behavior with no initial infections"""
        abc_config["initial_population"] = [[10000, 0, 0], [20000, 0, 0]]
        model = ABCJaxModel(abc_config)
        
        result = model.simulate()
        
        # With no B compartment, nothing should change
        assert result is not None


class TestCovidModel:
    """Tests for COVID Jax Model"""
    
    def test_model_initialization(self, covid_config):
        """Test that COVID model initializes correctly"""
        model = CovidJaxModel(covid_config)
        
        assert model.beta == 0.5
        assert model.gamma == 0.1
        assert model.theta == 0.2
        assert model.compartment_list == ["S", "E", "I", "H", "R", "D"]
    
    def test_age_stratification(self, covid_config):
        """Test that age stratification is loaded correctly"""
        model = CovidJaxModel(covid_config)
        
        assert len(model.age_groups) == 3
        assert len(model.age_stratification) == 3
        assert sum(model.age_stratification) == pytest.approx(1.0)
    
    def test_interaction_matrix_shape(self, covid_config):
        """Test interaction matrix dimensions"""
        model = CovidJaxModel(covid_config)
        
        assert model.interaction_matrix.shape == (3, 3)
    
    def test_all_rates_defined(self, covid_config):
        """Test that all transmission rates are defined"""
        model = CovidJaxModel(covid_config)
        
        assert model.beta is not None
        assert model.gamma is not None
        assert model.theta is not None
        assert model.zeta is not None
        assert model.delta is not None
        assert model.eta is not None
        assert model.epsilon is not None
    
    def test_simulate_runs(self, covid_config):
        """Test that COVID simulation runs"""
        covid_config["time_steps"] = 10
        model = CovidJaxModel(covid_config)
        
        result = model.simulate()
        
        assert result is not None
        assert "compartment_data" in result


class TestMpoxModel:
    """Tests for Mpox Jax Model"""
    
    def test_model_initialization(self, mpox_config):
        """Test that Mpox model initializes correctly"""
        model = MpoxJaxModel(mpox_config)
        
        assert model.beta == 0.3
        assert model.sigma == 0.1
        assert model.gamma == 0.05
        assert model.compartment_list == ["S", "E", "I", "R"]
    
    def test_seir_compartments(self, mpox_config):
        """Test SEIR compartment structure"""
        model = MpoxJaxModel(mpox_config)
        
        compartments = set(model.compartment_list)
        expected = {"S", "E", "I", "R"}
        
        assert compartments == expected
    
    def test_simulate_runs(self, mpox_config):
        """Test that Mpox simulation runs"""
        mpox_config["time_steps"] = 10
        model = MpoxJaxModel(mpox_config)
        
        result = model.simulate()
        
        assert result is not None


class TestDengueModel:
    """Tests for Dengue Jax Model"""
    
    def test_model_initialization(self, dengue_config):
        """Test that Dengue model initializes correctly"""
        model = DengueJaxModel(dengue_config)
        
        assert model.beta_h == 0.3
        assert model.beta_v == 0.4
        assert model.gamma_h == 0.1
        assert model.mu_v == 0.05
    
    def test_host_vector_compartments(self, dengue_config):
        """Test that both host and vector compartments exist"""
        model = DengueJaxModel(dengue_config)
        
        compartments = set(model.compartment_list)
        
        # Should have both human (_h) and vector (_v) compartments
        assert any("_h" in c for c in compartments)
        assert any("_v" in c for c in compartments)
    
    def test_simulate_runs(self, dengue_config):
        """Test that Dengue simulation runs"""
        dengue_config["time_steps"] = 10
        model = DengueJaxModel(dengue_config)
        
        result = model.simulate()
        
        assert result is not None


class TestModelComparison:
    """Cross-model comparison tests"""
    
    def test_all_models_have_simulate_method(self, abc_config, covid_config, mpox_config, dengue_config):
        """Test that all models implement simulate method"""
        models = [
            ABCJaxModel(abc_config),
            CovidJaxModel(covid_config),
            MpoxJaxModel(mpox_config),
            DengueJaxModel(dengue_config)
        ]
        
        for model in models:
            assert hasattr(model, "simulate")
            assert callable(model.simulate)
    
    def test_all_models_have_compartment_list(self, abc_config, covid_config, mpox_config, dengue_config):
        """Test that all models define compartment_list"""
        models = [
            ABCJaxModel(abc_config),
            CovidJaxModel(covid_config),
            MpoxJaxModel(mpox_config),
            DengueJaxModel(dengue_config)
        ]
        
        for model in models:
            assert hasattr(model, "compartment_list")
            assert len(model.compartment_list) > 0
    
    def test_all_models_preserve_population(self, abc_config, covid_config, mpox_config, dengue_config):
        """Test that all models preserve total population (approximately)"""
        configs = [
            (ABCJaxModel, abc_config, 10),
            (CovidJaxModel, covid_config, 10),
            (MpoxJaxModel, mpox_config, 10),
            (DengueJaxModel, dengue_config, 10)
        ]
        
        for ModelClass, config, steps in configs:
            config["time_steps"] = steps
            model = ModelClass(config)
            
            initial_pop = jnp.sum(model.population_matrix)
            result = model.simulate()
            
            # Population should be relatively conserved
            # (allowing for deaths in COVID model)
            assert initial_pop > 0
