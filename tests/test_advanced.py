"""Advanced tests for edge cases, numerical stability, and integration."""

import pytest
import jax.numpy as jnp
import jax
import numpy as np


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_precipitation(self):
        """Model should handle zero precipitation gracefully."""
        from jfuse.fuse import FUSEModel, Parameters, PRMS_CONFIG

        n_days = 50
        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)
        params = Parameters.default(n_hrus=1)

        precip = jnp.zeros(n_days)
        pet = jnp.ones(n_days) * 3.0
        temp = jnp.ones(n_days) * 15.0

        runoff, final_state = model.simulate((precip, pet, temp), params)

        assert jnp.all(jnp.isfinite(runoff))
        assert jnp.all(runoff >= 0)
        # Runoff should decrease over time with no input
        assert float(runoff[-1]) <= float(runoff[0]) + 0.1

    def test_extreme_precipitation(self):
        """Model should handle extreme precipitation events."""
        from jfuse.fuse import FUSEModel, Parameters, PRMS_CONFIG

        n_days = 10
        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)
        params = Parameters.default(n_hrus=1)

        # Extreme rain event (500 mm/day)
        precip = jnp.array([500.0] + [5.0] * (n_days - 1))
        pet = jnp.ones(n_days) * 3.0
        temp = jnp.ones(n_days) * 15.0

        runoff, final_state = model.simulate((precip, pet, temp), params)

        assert jnp.all(jnp.isfinite(runoff))
        assert jnp.all(runoff >= 0)
        # Should have high runoff on extreme day
        assert float(runoff[0]) > 10.0

    def test_freezing_temperatures(self):
        """Snow module should accumulate snow at freezing temps."""
        from jfuse.fuse import FUSEModel, Parameters, PRMS_CONFIG

        n_days = 30
        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)
        params = Parameters.default(n_hrus=1)

        precip = jnp.ones(n_days) * 10.0
        pet = jnp.ones(n_days) * 1.0
        temp = jnp.ones(n_days) * -10.0  # Well below freezing

        runoff, final_state = model.simulate((precip, pet, temp), params)

        assert jnp.all(jnp.isfinite(runoff))
        # Should have snow accumulation
        assert float(final_state.SWE) > 0

    def test_empty_storage_recovery(self):
        """Model should recover from near-empty storage."""
        from jfuse.fuse import FUSEModel, State, Parameters, PRMS_CONFIG

        n_days = 50
        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)
        params = Parameters.default(n_hrus=1)

        # Start with nearly empty storage
        initial_state = State(
            S1=jnp.array(0.1),
            S1_T=jnp.array(0.05),
            S1_TA=jnp.array(0.025),
            S1_TB=jnp.array(0.025),
            S1_F=jnp.array(0.05),
            S2=jnp.array(0.1),
            S2_T=jnp.array(0.05),
            S2_FA=jnp.array(0.025),
            S2_FB=jnp.array(0.025),
            SWE=jnp.array(0.0),
        )

        precip = jnp.ones(n_days) * 10.0
        pet = jnp.ones(n_days) * 2.0
        temp = jnp.ones(n_days) * 15.0

        runoff, final_state = model.simulate(
            (precip, pet, temp), params, initial_state=initial_state
        )

        assert jnp.all(jnp.isfinite(runoff))
        # Storage should increase
        assert float(final_state.S1) > float(initial_state.S1)


class TestWaterBalance:
    """Tests for water balance conservation."""

    def test_mass_conservation_no_snow(self):
        """Check approximate water balance without snow."""
        from jfuse.fuse import FUSEModel, State, Parameters, Forcing, PRMS_CONFIG

        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)
        params = Parameters.default(n_hrus=1)
        state = State.default(n_hrus=1)

        # Single timestep
        forcing = Forcing(
            precip=jnp.array(10.0),
            pet=jnp.array(3.0),
            temp=jnp.array(20.0),  # Warm - no snow
        )

        new_state, flux = model.step(state, forcing, params)

        # Water balance: dS = P - E - Q
        dS1 = float(new_state.S1) - float(state.S1)
        dS2 = float(new_state.S2) - float(state.S2)
        total_dS = dS1 + dS2

        P = float(forcing.precip)
        E = float(flux.e1) + float(flux.e2)
        Q = float(flux.q_total)

        balance_error = abs(total_dS - (P - E - Q))

        # Allow some error due to smooth approximations
        assert balance_error < 1.0, f"Water balance error: {balance_error}"

    def test_sacramento_free_split_honors_f_base(self):
        """Sacramento free-water recharge must split by f_base, not 50/50.

        The two parallel free reservoirs have capacities f_base*S2_F_max and
        (1-f_base)*S2_F_max, so their recharge must follow the same
        f_base:(1-f_base) ratio.
        """
        import equinox as eqx
        from jfuse.fuse import fuse_step, State, Parameters, Forcing
        from jfuse import SACRAMENTO_CONFIG

        f_base = 0.25
        params = Parameters.default(n_hrus=1)
        # Isolate the split: no baseflow drainage (v_A=v_B=0) and caps large
        # enough that neither free reservoir clamps over the test window.
        params = eqx.tree_at(
            lambda p: (p.f_base, p.v_A, p.v_B, p.S2_FA_max, p.S2_FB_max),
            params,
            (jnp.float32(f_base), jnp.float32(0.0), jnp.float32(0.0),
             jnp.float32(1e6), jnp.float32(1e6)),
        )

        state = State.default(n_hrus=1)
        state = eqx.tree_at(
            lambda s: (s.S2_FA, s.S2_FB), state,
            (jnp.float32(0.0), jnp.float32(0.0)),
        )

        forcing = Forcing(
            precip=jnp.float32(40.0), pet=jnp.float32(0.0), temp=jnp.float32(15.0)
        )

        # Accumulate recharge over several steps (no drainage, no clamp).
        for _ in range(15):
            state, _ = fuse_step(state, forcing, params, SACRAMENTO_CONFIG, 1.0, 1)

        fa = float(state.S2_FA)
        fb = float(state.S2_FB)
        assert fa + fb > 0.0, "no free-water recharge accumulated"
        # Split tracks f_base (~0.25), decisively not the old 50/50. Tolerance
        # accommodates float32 + smooth-clamp accumulation over the window.
        assert abs(fa / (fa + fb) - f_base) < 1e-2, (
            f"split {fa / (fa + fb):.3f} != f_base {f_base}"
        )

    def test_runoff_bounded_by_input(self):
        """Total runoff over time shouldn't wildly exceed input."""
        from jfuse.fuse import FUSEModel, Parameters, PRMS_CONFIG

        n_days = 365
        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)
        params = Parameters.default(n_hrus=1)

        total_precip = 1000.0
        precip = jnp.ones(n_days) * (total_precip / n_days)
        pet = jnp.ones(n_days) * 2.0
        temp = jnp.ones(n_days) * 15.0

        runoff, _ = model.simulate((precip, pet, temp), params)

        total_runoff = float(jnp.sum(runoff))

        # Runoff shouldn't exceed precip by more than initial storage
        # (allowing for some storage release)
        assert total_runoff < total_precip + 500  # 500mm buffer for initial storage


class TestJITCompilation:
    """Tests for JIT compilation compatibility."""

    def test_fuse_step_jittable(self):
        """fuse_step should be JIT compilable."""
        from jfuse.fuse import fuse_step, State, Parameters, Forcing, PRMS_CONFIG

        @jax.jit
        def jitted_step(state, forcing, params):
            return fuse_step(state, forcing, params, PRMS_CONFIG)

        state = State.default(n_hrus=1)
        params = Parameters.default(n_hrus=1)
        forcing = Forcing(
            precip=jnp.array(10.0),
            pet=jnp.array(3.0),
            temp=jnp.array(15.0),
        )

        # Should compile and run without error
        new_state, flux = jitted_step(state, forcing, params)
        assert jnp.isfinite(flux.q_total)

    def test_simulate_jittable(self):
        """Full simulation should be JIT compilable."""
        from jfuse.fuse import FUSEModel, Parameters, PRMS_CONFIG

        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)

        @jax.jit
        def jitted_simulate(precip, pet, temp, params):
            return model.simulate((precip, pet, temp), params)

        n_days = 30
        precip = jnp.ones(n_days) * 5.0
        pet = jnp.ones(n_days) * 3.0
        temp = jnp.ones(n_days) * 15.0
        params = Parameters.default(n_hrus=1)

        runoff, _ = jitted_simulate(precip, pet, temp, params)
        assert jnp.all(jnp.isfinite(runoff))

    def test_loss_jittable(self):
        """Loss computation should be JIT compilable."""
        from jfuse import nse_loss, kge_loss

        @jax.jit
        def jitted_losses(sim, obs):
            return nse_loss(sim, obs), kge_loss(sim, obs)

        sim = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        obs = jnp.array([1.1, 2.1, 2.9, 4.1, 4.9])

        nse, kge = jitted_losses(sim, obs)
        assert jnp.isfinite(nse)
        assert jnp.isfinite(kge)


class TestVectorization:
    """Tests for batched/vectorized operations."""

    def test_multi_hru_simulation(self):
        """Test simulation with multiple HRUs."""
        from jfuse.fuse import FUSEModel, Parameters, PRMS_CONFIG

        n_days = 30
        n_hrus = 5
        model = FUSEModel(config=PRMS_CONFIG, n_hrus=n_hrus)
        params = Parameters.default(n_hrus=n_hrus)

        # Different forcing for each HRU
        precip = jnp.ones((n_days, n_hrus)) * jnp.array([5, 10, 15, 20, 25])
        pet = jnp.ones((n_days, n_hrus)) * 3.0
        temp = jnp.ones((n_days, n_hrus)) * 15.0

        runoff, final_state = model.simulate((precip, pet, temp), params)

        assert runoff.shape == (n_days, n_hrus)
        assert jnp.all(jnp.isfinite(runoff))

        # Higher precip should give higher runoff on average
        mean_runoff = jnp.mean(runoff, axis=0)
        assert float(mean_runoff[-1]) > float(mean_runoff[0])

    def test_vmap_over_parameters(self):
        """Test vmapping simulation over parameter sets."""
        from jfuse.fuse import FUSEModel, Parameters, PRMS_CONFIG

        n_days = 20
        n_param_sets = 3
        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)

        precip = jnp.ones(n_days) * 10.0
        pet = jnp.ones(n_days) * 3.0
        temp = jnp.ones(n_days) * 15.0

        # Create multiple parameter sets with different S1_max
        base_params = Parameters.default(n_hrus=1)

        def simulate_with_s1max(s1_max):
            # Create modified params (simplified - just vary one param)
            import equinox as eqx
            params = eqx.tree_at(lambda p: p.S1_max, base_params, s1_max)
            runoff, _ = model.simulate((precip, pet, temp), params)
            return jnp.mean(runoff)

        s1_max_values = jnp.array([100.0, 200.0, 300.0])
        mean_runoffs = jax.vmap(simulate_with_s1max)(s1_max_values)

        assert mean_runoffs.shape == (n_param_sets,)
        assert jnp.all(jnp.isfinite(mean_runoffs))


class TestGradientNumerics:
    """Tests for gradient computation accuracy."""

    def test_gradient_finite_difference_check(self):
        """Compare autodiff gradients to finite differences."""
        from jfuse.fuse import FUSEModel, Parameters, PRMS_CONFIG
        import equinox as eqx

        n_days = 20
        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)
        params = Parameters.default(n_hrus=1)

        precip = jnp.ones(n_days) * 10.0
        pet = jnp.ones(n_days) * 3.0
        temp = jnp.ones(n_days) * 15.0

        def loss_fn(s1_max):
            p = eqx.tree_at(lambda p: p.S1_max, params, s1_max)
            runoff, _ = model.simulate((precip, pet, temp), p)
            return jnp.mean(runoff)

        s1_max = jnp.array(200.0)

        # Autodiff gradient
        auto_grad = jax.grad(loss_fn)(s1_max)

        # Finite difference gradient
        eps = 1e-4
        fd_grad = (loss_fn(s1_max + eps) - loss_fn(s1_max - eps)) / (2 * eps)

        # Should be reasonably close
        rel_error = abs(float(auto_grad) - float(fd_grad)) / (abs(float(fd_grad)) + 1e-8)
        assert rel_error < 0.1, f"Gradient mismatch: autodiff={auto_grad}, fd={fd_grad}"

    def test_no_nan_gradients(self):
        """Gradients should not be NaN for reasonable inputs."""
        from jfuse.fuse import FUSEModel, Parameters, PRMS_CONFIG

        n_days = 30
        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)

        precip = jnp.ones(n_days) * 10.0
        pet = jnp.ones(n_days) * 3.0
        temp = jnp.ones(n_days) * 15.0

        def loss_fn(params):
            runoff, _ = model.simulate((precip, pet, temp), params)
            return jnp.mean(runoff ** 2)

        params = Parameters.default(n_hrus=1)
        grads = jax.grad(loss_fn)(params)

        # Check no NaN gradients
        grad_leaves = jax.tree_util.tree_leaves(grads)
        for g in grad_leaves:
            assert jnp.all(jnp.isfinite(g)), "NaN gradient detected"


class TestCoupledModel:
    """Tests for coupled FUSE + routing model."""

    def test_coupled_simulation(self):
        """Test end-to-end coupled simulation."""
        from jfuse import CoupledModel, PRMS_CONFIG
        from jfuse.routing import create_network_from_topology

        # Create simple network
        reach_ids = [0, 1, 2]
        downstream_ids = [1, 2, -1]
        lengths = [1000.0, 2000.0, 1500.0]
        slopes = [0.01, 0.005, 0.002]

        network = create_network_from_topology(
            reach_ids, downstream_ids, lengths, slopes
        )

        n_hrus = 3
        hru_areas = jnp.ones(n_hrus) * 1e6  # 1 km² each

        model = CoupledModel(
            fuse_config=PRMS_CONFIG,
            network=network.to_arrays(),
            hru_areas=hru_areas,
            n_hrus=n_hrus,
        )

        n_days = 30
        precip = jnp.ones((n_days, n_hrus)) * 10.0
        pet = jnp.ones((n_days, n_hrus)) * 3.0
        temp = jnp.ones((n_days, n_hrus)) * 15.0

        params = model.default_params()
        outlet_Q, runoff = model.simulate((precip, pet, temp), params)

        assert outlet_Q.shape == (n_days,)
        assert runoff.shape == (n_days, n_hrus)
        assert jnp.all(jnp.isfinite(outlet_Q))
        assert jnp.all(outlet_Q >= 0)

    def test_coupled_gradient_flow(self):
        """Test gradient flow through coupled model."""
        from jfuse import CoupledModel, PRMS_CONFIG
        from jfuse.routing import create_network_from_topology

        reach_ids = [0, 1]
        downstream_ids = [1, -1]
        lengths = [1000.0, 1500.0]
        slopes = [0.01, 0.005]

        network = create_network_from_topology(
            reach_ids, downstream_ids, lengths, slopes
        )

        n_hrus = 2
        hru_areas = jnp.ones(n_hrus) * 1e6

        model = CoupledModel(
            fuse_config=PRMS_CONFIG,
            network=network.to_arrays(),
            hru_areas=hru_areas,
            n_hrus=n_hrus,
        )

        n_days = 20
        precip = jnp.ones((n_days, n_hrus)) * 10.0
        pet = jnp.ones((n_days, n_hrus)) * 3.0
        temp = jnp.ones((n_days, n_hrus)) * 15.0

        def loss_fn(params):
            outlet_Q, _ = model.simulate((precip, pet, temp), params)
            return jnp.mean(outlet_Q)

        params = model.default_params()
        loss, grads = jax.value_and_grad(loss_fn)(params)

        assert jnp.isfinite(loss)
        # Check FUSE param gradients (may be array for multiple HRUs)
        assert jnp.all(jnp.isfinite(grads.fuse_params.S1_max))
        # Check routing param gradients
        assert jnp.all(jnp.isfinite(grads.manning_n))

    def test_routing_dt_defaults_to_fuse_step(self):
        """Daily FUSE coupling must route at the FUSE step interval, not 3600s.

        The router advances one step per inflow row; routing a daily series at
        a smaller dt (with no sub-stepping) over-attenuates/over-lags the
        hydrograph. The default routing_dt must therefore equal fuse_dt*86400.
        """
        from jfuse.coupled import coupled_simulate
        from jfuse import PRMS_CONFIG
        from jfuse.fuse import Parameters
        from jfuse.routing import create_network_from_topology

        # Long reaches make the dt sensitivity pronounced.
        n_hrus = 2
        network = create_network_from_topology(
            [0, 1], [1, -1], [50000.0, 50000.0], [0.001, 0.001]
        ).to_arrays()
        fuse_params = Parameters.default(n_hrus=n_hrus)
        hru_areas = jnp.ones(n_hrus) * 1e7

        n_days = 30
        forcing = (
            jnp.ones((n_days, n_hrus)) * 10.0,
            jnp.ones((n_days, n_hrus)) * 3.0,
            jnp.ones((n_days, n_hrus)) * 15.0,
        )

        def run(routing_dt):
            out, _, _ = coupled_simulate(
                forcing, fuse_params, network.manning_n, network, hru_areas,
                PRMS_CONFIG, fuse_dt=1.0, routing_dt=routing_dt,
            )
            return out

        out_default = run(None)
        out_daily = run(86400.0)
        out_hourly = run(3600.0)

        # Default resolves to the daily step ...
        assert jnp.allclose(out_default, out_daily)
        # ... and is distinct from the previous (buggy) 3600s behavior.
        assert not jnp.allclose(out_default, out_hourly, atol=1e-4)

    def test_single_hru_coupled_runs(self):
        """A single HRU + single reach coupled run must not raise (regression).

        Previously the scalar default FUSE state (n_hrus=1) paired with
        shape-(1,) forcing rows tripped a lax.scan carry-type mismatch.
        """
        from jfuse import CoupledModel, PRMS_CONFIG
        from jfuse.routing import create_network_from_topology

        network = create_network_from_topology([0], [-1], [1000.0], [0.01])
        model = CoupledModel(
            fuse_config=PRMS_CONFIG,
            network=network.to_arrays(),
            hru_areas=jnp.ones(1) * 1e6,
            n_hrus=1,
        )

        n_days = 20
        precip = jnp.ones((n_days, 1)) * 10.0
        pet = jnp.ones((n_days, 1)) * 3.0
        temp = jnp.ones((n_days, 1)) * 15.0

        outlet_Q, runoff = model.simulate((precip, pet, temp), model.default_params())

        assert outlet_Q.shape == (n_days,)
        assert runoff.shape == (n_days, 1)
        assert jnp.all(jnp.isfinite(outlet_Q))
        assert jnp.all(outlet_Q >= 0)

    def test_routing_substeps_conserve_and_resolve(self):
        """Sub-stepping stays finite/mass-conserving and resolves sane counts."""
        from jfuse import CoupledModel, PRMS_CONFIG
        from jfuse.routing import create_network_from_topology, route_network
        from jfuse.coupled import _resolve_n_substeps

        net = create_network_from_topology([0], [-1], [2000.0], [0.005]).to_arrays()
        n_t = 60
        inflow = jnp.ones((n_t, 1)) * 5.0

        out1 = route_network(inflow, net, dt=86400.0, n_substeps=1)
        out8 = route_network(inflow, net, dt=86400.0, n_substeps=8)

        assert out8.shape == (n_t,)
        assert jnp.all(jnp.isfinite(out8))
        # Steady-state outlet equals the constant inflow, independent of sub-steps.
        assert abs(float(out8[-1]) - 5.0) < 1e-2
        assert abs(float(out1[-1]) - 5.0) < 1e-2

        # Resolver: fixed honors the count, max=1 disables, adaptive stays bounded.
        assert _resolve_n_substeps('fixed', 5, net, 86400.0) == 5
        assert _resolve_n_substeps('adaptive', 1, net, 86400.0) == 1
        assert 1 <= _resolve_n_substeps('adaptive', 10, net, 86400.0) <= 10

        # CoupledModel exposes the resolved static count.
        common = dict(
            fuse_config=PRMS_CONFIG, network=net,
            hru_areas=jnp.ones(1) * 1e6, n_hrus=1,
        )
        assert CoupledModel(**common, routing_max_substeps=1).n_substeps == 1
        assert CoupledModel(
            **common, routing_substep_method='fixed', routing_max_substeps=4
        ).n_substeps == 4


class TestLongSimulations:
    """Tests for numerical stability over long simulations."""

    def test_multi_year_stability(self):
        """Test stability over multi-year simulation."""
        from jfuse.fuse import FUSEModel, Parameters, PRMS_CONFIG

        n_days = 365 * 3  # 3 years
        model = FUSEModel(config=PRMS_CONFIG, n_hrus=1)
        params = Parameters.default(n_hrus=1)

        # Seasonal forcing pattern
        t = jnp.arange(n_days)
        precip = 5.0 + 5.0 * jnp.sin(2 * jnp.pi * t / 365)
        pet = 2.0 + 2.0 * jnp.sin(2 * jnp.pi * (t - 90) / 365)
        temp = 10.0 + 15.0 * jnp.sin(2 * jnp.pi * (t - 90) / 365)

        runoff, final_state = model.simulate((precip, pet, temp), params)

        # No NaN or Inf
        assert jnp.all(jnp.isfinite(runoff))
        assert jnp.all(jnp.isfinite(final_state.S1))
        assert jnp.all(jnp.isfinite(final_state.S2))

        # Storage should remain bounded
        assert float(final_state.S1) < float(params.S1_max) * 1.1
        assert float(final_state.S2) < float(params.S2_max) * 1.1
        assert float(final_state.S1) >= 0
        assert float(final_state.S2) >= 0
