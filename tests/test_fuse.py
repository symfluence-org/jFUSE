"""Tests for FUSE model core functionality."""

import jax.numpy as jnp
import jax


class TestState:
    """Tests for State class."""

    def test_default_state_creation(self):
        """Test creating default state."""
        from jfuse.fuse import State

        state = State.default(n_hrus=1)
        assert state.S1.shape == ()
        assert float(state.S1) > 0

    def test_default_state_batched(self):
        """Test creating batched default state."""
        from jfuse.fuse import State

        state = State.default(n_hrus=5)
        assert state.S1.shape == (5,)
        assert state.S2.shape == (5,)
        assert state.SWE.shape == (5,)

    def test_state_to_array(self):
        """Test state serialization to array."""
        from jfuse.fuse import State

        state = State.default(n_hrus=1)
        arr = state.to_array()
        # 10 soil/snow stores + 3 glacier states (ICE, S_glac, SWE_glac)
        assert arr.shape == (13,)

    def test_state_roundtrip(self):
        """Test state array roundtrip."""
        from jfuse.fuse import State

        state = State.default(n_hrus=1)
        arr = state.to_array()
        state2 = State.from_array(arr)

        assert jnp.allclose(state.S1, state2.S1)
        assert jnp.allclose(state.S2, state2.S2)


class TestParameters:
    """Tests for Parameters class."""

    def test_default_parameters(self):
        """Test creating default parameters."""
        from jfuse.fuse import Parameters

        params = Parameters.default(n_hrus=1)
        assert params.S1_max.shape == ()
        assert float(params.S1_max) > 0

    def test_parameters_to_array(self):
        """Test parameter serialization."""
        from jfuse.fuse import Parameters, PARAM_NAMES

        params = Parameters.default(n_hrus=1)
        arr = params.to_array()
        assert arr.shape == (len(PARAM_NAMES),)

    def test_parameters_roundtrip(self):
        """Test parameter array roundtrip."""
        from jfuse.fuse import Parameters

        params = Parameters.default(n_hrus=1)
        arr = params.to_array()
        params2 = Parameters.from_array(arr, n_hrus=1)

        assert jnp.allclose(params.S1_max, params2.S1_max)
        assert jnp.allclose(params.ku, params2.ku)

    def test_derived_parameters_computed(self):
        """Test that derived parameters are computed correctly."""
        from jfuse.fuse import Parameters

        params = Parameters.default(n_hrus=1)

        # Check derived parameter relationships
        expected_S1_T_max = params.f_tens * params.S1_max
        assert jnp.allclose(params.S1_T_max, expected_S1_T_max)

        expected_S1_F_max = (1.0 - params.f_tens) * params.S1_max
        assert jnp.allclose(params.S1_F_max, expected_S1_F_max)

    def test_validate_bounds(self):
        """Test parameter bounds validation."""
        from jfuse.fuse import Parameters
        import warnings

        params = Parameters.default(n_hrus=1)

        # Default params should be valid
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert params.validate_bounds(warn=True)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_predefined_configs(self):
        """Test predefined model configurations."""
        from jfuse.fuse import PRMS_CONFIG, SACRAMENTO_CONFIG, TOPMODEL_CONFIG, VIC_CONFIG

        assert PRMS_CONFIG.num_states > 0
        assert SACRAMENTO_CONFIG.num_states > 0
        assert TOPMODEL_CONFIG.num_states > 0
        assert VIC_CONFIG.num_states > 0

    def test_config_describe(self):
        """Test config description."""
        from jfuse.fuse import PRMS_CONFIG

        desc = PRMS_CONFIG.describe()
        assert "PRMS" in desc or "TENSION2_FREE" in desc

    def test_get_config(self):
        """Test getting config by name."""
        from jfuse.fuse import get_config, PRMS_CONFIG

        config = get_config("prms")
        assert config == PRMS_CONFIG


class TestPhysics:
    """Tests for physics functions."""

    def test_smooth_functions_differentiable(self):
        """Test that smooth functions are differentiable."""
        from jfuse.fuse import physics

        x = jnp.array(0.5)

        # Test smooth_sigmoid
        grad_fn = jax.grad(lambda x: physics.smooth_sigmoid(x, k=1.0))
        grad = grad_fn(x)
        assert jnp.isfinite(grad)

        # Test smooth_max
        y = jnp.array(0.3)
        grad_fn = jax.grad(lambda x: physics.smooth_max(x, y, k=0.01))
        grad = grad_fn(x)
        assert jnp.isfinite(grad)

    def test_snow_partitioning(self):
        """Test snow rain/snow partitioning."""
        from jfuse.fuse import physics

        precip = jnp.array(10.0)
        SWE = jnp.array(50.0)
        T_rain = jnp.array(1.0)
        T_melt = jnp.array(0.0)
        melt_rate = jnp.array(3.0)

        # Cold temperature - should be snow
        rain, melt, SWE_new = physics.compute_snow(
            precip, jnp.array(-5.0), SWE, T_rain, T_melt, melt_rate
        )
        assert float(rain) < 1.0  # Most is snow
        assert float(melt) < 1.0  # Little melt

        # Warm temperature - should be rain with melt
        rain, melt, SWE_new = physics.compute_snow(
            precip, jnp.array(10.0), SWE, T_rain, T_melt, melt_rate
        )
        assert float(rain) > 5.0  # Most is rain
        assert float(melt) > 0.0  # Some melt

    def test_evaporation_sequential(self):
        """Test sequential evaporation."""
        from jfuse.fuse import physics

        pet = jnp.array(5.0)
        S1 = jnp.array(100.0)
        S2 = jnp.array(200.0)
        S1_max = jnp.array(200.0)
        S2_max = jnp.array(500.0)

        e1, e2 = physics.compute_evaporation_sequential(pet, S1, S2, S1_max, S2_max)

        # Upper layer should take priority
        assert float(e1) > 0
        assert float(e1) <= float(pet) + 0.01  # Small tolerance for smooth approximations
        # Total evap should approximately not exceed PET (small tolerance for smooth functions)
        assert float(e1) + float(e2) <= float(pet) + 0.01

    def test_baseflow_positive(self):
        """Test baseflow is always non-negative."""
        from jfuse.fuse import physics

        S2 = jnp.array(100.0)
        S2_max = jnp.array(500.0)
        v = jnp.array(0.1)
        ks = jnp.array(0.01)
        n = jnp.array(2.0)
        m = jnp.array(50.0)

        qb_linear = physics.compute_baseflow_linear(S2, v)
        qb_nonlinear = physics.compute_baseflow_nonlinear(S2, S2_max, ks, n)
        qb_topmodel = physics.compute_baseflow_topmodel(S2, S2_max, ks, m)

        assert float(qb_linear) >= 0
        assert float(qb_nonlinear) >= 0
        assert float(qb_topmodel) >= 0


class TestFUSEModel:
    """Tests for FUSE model."""

    def test_single_step(self):
        """Test single timestep computation."""
        from jfuse.fuse import FUSEModel, State, Parameters, Forcing, PRMS_CONFIG

        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)
        state = State.default(n_hrus=1)
        params = Parameters.default(n_hrus=1)
        forcing = Forcing(
            precip=jnp.array(10.0),
            pet=jnp.array(3.0),
            temp=jnp.array(15.0),
        )

        new_state, flux = model.step(state, forcing, params)

        # Check outputs are valid
        assert jnp.isfinite(new_state.S1)
        assert jnp.isfinite(flux.q_total)
        assert float(flux.q_total) >= 0

    def test_simulate(self):
        """Test full simulation."""
        from jfuse.fuse import FUSEModel, Parameters, PRMS_CONFIG

        n_days = 100
        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)
        params = Parameters.default(n_hrus=1)

        # Create synthetic forcing
        precip = jnp.ones(n_days) * 5.0
        pet = jnp.ones(n_days) * 3.0
        temp = jnp.ones(n_days) * 15.0

        runoff, final_state = model.simulate((precip, pet, temp), params)

        assert runoff.shape == (n_days,)
        assert jnp.all(jnp.isfinite(runoff))
        assert jnp.all(runoff >= 0)

    def test_different_configs(self):
        """Test model with different configurations."""
        from jfuse.fuse import (
            FUSEModel,
            Parameters,
            PRMS_CONFIG,
            SACRAMENTO_CONFIG,
            TOPMODEL_CONFIG,
            VIC_CONFIG,
        )

        n_days = 10
        configs = [PRMS_CONFIG, SACRAMENTO_CONFIG, TOPMODEL_CONFIG, VIC_CONFIG]

        precip = jnp.ones(n_days) * 5.0
        pet = jnp.ones(n_days) * 3.0
        temp = jnp.ones(n_days) * 15.0

        for config in configs:
            model = FUSEModel(config=config, n_hrus=1)
            params = Parameters.default(n_hrus=1)
            runoff, _ = model.simulate((precip, pet, temp), params)

            assert runoff.shape == (n_days,)
            assert jnp.all(jnp.isfinite(runoff)), f"NaN in {config}"

    def test_gradient_flow(self):
        """Test that gradients flow through simulation."""
        from jfuse.fuse import FUSEModel, Parameters, PRMS_CONFIG

        n_days = 10
        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)

        precip = jnp.ones(n_days) * 5.0
        pet = jnp.ones(n_days) * 3.0
        temp = jnp.ones(n_days) * 15.0

        def loss_fn(params):
            runoff, _ = model.simulate((precip, pet, temp), params)
            return jnp.mean(runoff)

        params = Parameters.default(n_hrus=1)
        loss, grads = jax.value_and_grad(loss_fn)(params)

        assert jnp.isfinite(loss)
        assert jnp.isfinite(grads.S1_max)
        assert jnp.isfinite(grads.ku)


class TestRouting:
    """Tests for river routing."""

    def test_network_creation(self):
        """Test network creation."""
        from jfuse.routing import RiverNetwork, Reach

        network = RiverNetwork()
        network.add_reach(Reach(id=0, length=1000, slope=0.001))
        network.add_reach(
            Reach(id=1, length=2000, slope=0.0005, upstream_ids=[0], downstream_id=-1)
        )
        network.build_topology()

        assert network.n_reaches == 2
        assert 0 in network.topological_order
        assert 1 in network.topological_order

    def test_network_arrays(self):
        """Test conversion to arrays."""
        from jfuse.routing import create_network_from_topology

        reach_ids = [0, 1, 2]
        downstream_ids = [1, 2, -1]
        lengths = [1000.0, 2000.0, 1500.0]
        slopes = [0.01, 0.005, 0.002]

        network = create_network_from_topology(reach_ids, downstream_ids, lengths, slopes)
        arrays = network.to_arrays()

        assert arrays.n_reaches == 3
        assert len(arrays.lengths) == 3

    def test_muskingum_params(self):
        """Test Muskingum parameter computation."""
        from jfuse.routing import compute_muskingum_params

        Q = jnp.array(10.0)
        length = jnp.array(1000.0)
        slope = jnp.array(0.001)
        manning_n = jnp.array(0.035)
        width_coef = jnp.array(7.2)
        width_exp = jnp.array(0.5)
        depth_coef = jnp.array(0.27)
        depth_exp = jnp.array(0.3)
        dt = 3600.0

        params = compute_muskingum_params(
            Q, length, slope, manning_n, width_coef, width_exp, depth_coef, depth_exp, dt
        )

        # Check coefficients sum to 1
        total = params.C0 + params.C1 + params.C2
        assert jnp.isclose(total, 1.0, atol=1e-5)

        # Check all positive
        assert float(params.C0) >= 0
        assert float(params.C1) >= 0
        assert float(params.C2) >= 0


class TestLossFunctions:
    """Tests for loss functions."""

    def test_nse_loss(self):
        """Test NSE loss computation."""
        from jfuse import nse_loss

        sim = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        obs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        loss = nse_loss(sim, obs)
        assert float(loss) < 0.01  # Perfect match

        # Bad simulation
        sim_bad = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])
        loss_bad = nse_loss(sim_bad, obs)
        assert float(loss_bad) > float(loss)

    def test_kge_loss(self):
        """Test KGE loss computation."""
        from jfuse import kge_loss

        sim = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        obs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        loss = kge_loss(sim, obs)
        assert float(loss) < 0.01  # Perfect match

    def test_loss_with_nan(self):
        """Test loss functions handle NaN observations."""
        from jfuse import nse_loss

        sim = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        obs = jnp.array([1.0, jnp.nan, 3.0, jnp.nan, 5.0])

        loss = nse_loss(sim, obs)
        assert jnp.isfinite(loss)


class TestCalibration:
    """Tests for calibration utilities."""

    def test_transform_roundtrip(self):
        """Test bounded<->unbounded transform roundtrip."""
        from jfuse.fuse import Parameters
        from jfuse.optim.calibration import transform_to_unbounded, transform_to_bounded

        params = Parameters.default(n_hrus=1)
        unbounded = transform_to_unbounded(params)
        recovered = transform_to_bounded(unbounded)

        # Check key parameters are recovered
        assert jnp.isclose(params.S1_max, recovered.S1_max, rtol=1e-4)
        assert jnp.isclose(params.ku, recovered.ku, rtol=1e-4)

    def test_optimizer_creation(self):
        """Test optimizer creation."""
        from jfuse.optim.calibration import create_optimizer, CalibrationConfig

        config = CalibrationConfig(max_iterations=100, learning_rate=0.01, optimizer="adam")
        optimizer = create_optimizer(config)

        # Should create without error
        assert optimizer is not None


class TestImports:
    """Test that all public exports are importable."""

    def test_main_imports(self):
        """Test main package imports."""

    def test_subpackage_imports(self):
        """Test subpackage imports."""
