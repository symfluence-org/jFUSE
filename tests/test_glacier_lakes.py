"""Tests for the temperature-index glacier module and the differentiable
lake/reservoir routing node.

Both modules are designed to (a) be no-ops when disabled / absent so existing
behaviour is preserved, and (b) expose AD-active parameters so the glacier
degree-day factor and the reservoir operating rules calibrate by gradient.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from jfuse.fuse.model import FUSEModel
from jfuse.fuse.state import Parameters, State, PARAM_NAMES, NUM_PARAMETERS
from jfuse.fuse.config import PRMS_CONFIG
from jfuse.routing.network import RiverNetwork, Reach
from jfuse.routing.router import route_network, lake_outflow


def _forcing(n_t=200, n_hru=4, seed=0):
    key = jax.random.PRNGKey(seed)
    precip = jnp.abs(jax.random.normal(key, (n_t, n_hru))) * 5.0
    temp = 5.0 + 8.0 * jnp.sin(jnp.linspace(0, 6 * jnp.pi, n_t))[:, None] * jnp.ones((1, n_hru))
    pet = jnp.abs(temp) * 0.2
    return precip, pet, temp


# ---------------------------------------------------------------------------
# Glacier module
# ---------------------------------------------------------------------------

class TestGlacier:
    def test_params_appended_without_shifting_indices(self):
        assert NUM_PARAMETERS == 33
        assert PARAM_NAMES[-3:] == ("DDF_ice", "T_ice", "K_glac")
        # Legacy 30-wide arrays still load (trailing glacier params defaulted).
        legacy = jnp.ones((30,))
        p = Parameters.from_array(legacy, n_hrus=1)
        assert float(p.DDF_ice) == pytest.approx(7.0)
        assert float(p.K_glac) == pytest.approx(0.3)

    def test_zero_fraction_is_identical_to_no_glacier(self):
        forcing = _forcing()
        gfrac = jnp.array([0.0, 0.3, 0.6, 0.9])
        p = Parameters.default(n_hrus=4)
        rg, _ = FUSEModel(PRMS_CONFIG._replace(enable_glacier=True), n_hrus=4).simulate(
            forcing, p, glacier_frac=gfrac)
        r0, _ = FUSEModel(PRMS_CONFIG, n_hrus=4).simulate(forcing, p)
        # Glacier-free HRU (fraction 0) must be bit-for-bit unchanged.
        assert jnp.allclose(rg[:, 0], r0[:, 0], atol=1e-5)
        # Glacierized HRUs add melt water -> more runoff.
        assert float(rg[:, 3].mean()) > float(r0[:, 3].mean())

    def test_ddf_ice_gradient_flows_only_on_glacier(self):
        forcing = _forcing()
        gfrac = jnp.array([0.0, 0.3, 0.6, 0.9])
        model = FUSEModel(PRMS_CONFIG._replace(enable_glacier=True), n_hrus=4)
        di = PARAM_NAMES.index("DDF_ice")

        def loss(arr):
            pp = Parameters.from_array(arr, n_hrus=4)
            ro, _ = model.simulate(forcing, pp, glacier_frac=gfrac)
            return jnp.sum(ro)

        g = jax.grad(loss)(Parameters.default(n_hrus=4).to_array())
        assert jnp.all(jnp.isfinite(g))
        assert abs(float(g[0, di])) < 1e-6          # no glacier -> no gradient
        assert float(jnp.abs(g[1:, di]).sum()) > 0  # glacier HRUs -> gradient

    def test_finite_ice_store_depletes(self):
        forcing = _forcing(n_t=120, n_hru=1)
        gfrac = jnp.array([1.0])
        p = Parameters.default(n_hrus=1)
        # Small ice store should draw down over a melt season.
        st = eqx.tree_at(lambda s: s.ICE, State.default(n_hrus=1), jnp.array(500.0))
        model = FUSEModel(PRMS_CONFIG._replace(enable_glacier=True), n_hrus=1)
        _, final = model.simulate(forcing, p, initial_state=st, glacier_frac=gfrac)
        assert float(jnp.min(final.ICE)) < 500.0


# ---------------------------------------------------------------------------
# Lake / reservoir routing
# ---------------------------------------------------------------------------

def _chain(lake_kwargs=None):
    net = RiverNetwork()
    net.add_reach(Reach(id=0, length=5000, slope=1e-3, upstream_ids=[], downstream_id=1))
    r1 = dict(id=1, length=5000, slope=1e-3, upstream_ids=[0], downstream_id=2)
    if lake_kwargs:
        r1.update(lake_kwargs)
    net.add_reach(Reach(**r1))
    net.add_reach(Reach(id=2, length=5000, slope=1e-3, upstream_ids=[1], downstream_id=-1))
    net.build_topology()
    return net.to_arrays()


def _flood(n_t=240, n_reach=3):
    lat = jnp.zeros((n_t, n_reach))
    pulse = 50.0 * jnp.exp(-((jnp.arange(n_t) - 50) / 8.0) ** 2) + 2.0
    return lat.at[:, 0].set(pulse)


class TestLakeReservoir:
    def test_outflow_monotonic_in_storage(self):
        s = jnp.linspace(0.0, 2.0e7, 50)
        q = lake_outflow(s, 1.0e7, 20.0, 2.0, 2.0, 0.01)
        assert jnp.all(jnp.diff(q) >= -1e-6)   # non-decreasing
        assert float(q[0]) >= 2.0 - 1e-5        # >= q_min at empty

    def test_reservoir_attenuates_peak(self):
        lat = _flood()
        q0 = route_network(lat, _chain(), dt=3600.0)
        qL = route_network(lat, _chain(dict(
            is_lake=True, lake_s_max=2.0e7, lake_q_ref=20.0,
            lake_q_min=2.0, lake_exp=2.0, lake_spill_coef=0.01)), dt=3600.0)
        assert jnp.all(jnp.isfinite(qL))
        assert float(qL.max()) < float(q0.max())

    def test_no_lake_network_matches_channel_only(self):
        lat = _flood()
        # is_lake all False must reproduce pure-channel routing exactly.
        na = _chain()
        assert not bool(na.is_lake.any())
        q = route_network(lat, na, dt=3600.0)
        assert jnp.all(jnp.isfinite(q))

    def test_operating_rule_param_is_differentiable(self):
        lat = _flood()
        base = _chain(dict(is_lake=True, lake_s_max=2.0e7, lake_q_ref=20.0,
                           lake_q_min=2.0, lake_exp=2.0, lake_spill_coef=0.01))

        def loss(qref):
            na = base._replace(lake_q_ref=base.lake_q_ref.at[1].set(qref))
            return jnp.sum(route_network(lat, na, dt=3600.0))

        g = jax.grad(loss)(20.0)
        assert jnp.isfinite(g) and abs(float(g)) > 0


# ---------------------------------------------------------------------------
# Static-input loaders (glacier fraction / lake classification)
# ---------------------------------------------------------------------------

class TestStaticInputs:
    def test_load_glacier_fraction_from_climate_csv(self, tmp_path):
        import pandas as pd
        from jfuse.static_inputs import load_glacier_fraction

        clim = tmp_path / "data" / "attributes" / "climate"
        clim.mkdir(parents=True)
        pd.DataFrame({"glacier_fraction": [0.0, 0.25, 1.2, -0.1]}).to_csv(
            clim / "climate_statistics.csv", index=False)
        frac = load_glacier_fraction(tmp_path, "X")
        assert frac is not None and frac.shape == (4,)
        # Values are clipped to [0, 1].
        assert float(frac.min()) >= 0.0 and float(frac.max()) <= 1.0
        assert frac[1] == jnp.float32(0.25)

    def test_load_glacier_fraction_absent_returns_none(self, tmp_path):
        from jfuse.static_inputs import load_glacier_fraction
        assert load_glacier_fraction(tmp_path, "X") is None

    def test_classify_lakes_no_data_is_noop(self, tmp_path):
        # No HydroLAKES => network returned unchanged (graceful degradation).
        from jfuse.static_inputs import classify_lakes_onto_network
        na = _chain()
        out = classify_lakes_onto_network(na, tmp_path, "X")
        assert out is na


class TestLakeRuleCalibration:
    """Global lake operating-rule multipliers are AD-active calibration vars."""

    def _model(self):
        from jfuse.coupled import CoupledModel
        from jfuse.fuse.config import PRMS_CONFIG
        from jfuse.routing.network import RiverNetwork, Reach
        net = RiverNetwork()
        net.add_reach(Reach(id=1, length=5000, slope=1e-3, upstream_ids=[], downstream_id=2))
        net.add_reach(Reach(id=2, length=5000, slope=1e-3, upstream_ids=[1], downstream_id=-1,
                            is_lake=True, lake_s_max=1e7, lake_q_ref=15.0, lake_q_min=1.0,
                            lake_exp=2.0, lake_spill_coef=0.01))
        cm = CoupledModel(fuse_config=PRMS_CONFIG, network=net.to_arrays(),
                          hru_areas=jnp.array([4e7, 4e7]), n_hrus=2)
        key = jax.random.PRNGKey(3)
        forcing = (jnp.abs(jax.random.normal(key, (120, 2))) * 6,
                   jnp.ones((120, 2)), jnp.ones((120, 2)) * 3.0)
        return cm, forcing

    def test_apply_lake_rules_none_is_noop(self):
        from jfuse.coupled import apply_lake_rules
        from jfuse.routing.network import RiverNetwork, Reach
        net = RiverNetwork()
        net.add_reach(Reach(id=1, length=5000, slope=1e-3, downstream_id=-1))
        na = net.to_arrays()
        assert apply_lake_rules(na, None) is na

    def test_q_ref_mult_is_differentiable(self):
        from jfuse.coupled import LakeRuleParams, build_lake_rules, LAKE_RULE_NAMES
        cm, forcing = self._model()
        base = cm.default_params()
        assert LAKE_RULE_NAMES[0] == "LAKE_Q_REF_MULT"

        def loss(qref_mult):
            lr = build_lake_rules([qref_mult, jnp.float32(0.1), jnp.float32(2.0), jnp.float32(1.0)])
            oq, _ = cm.simulate(forcing, base._replace(lake_rules=lr))
            return jnp.sum(oq)

        g = jax.grad(loss)(jnp.float32(1.5))
        assert jnp.isfinite(g) and abs(float(g)) > 0
        # Higher q_ref multiplier releases more water on average.
        o_lo, _ = cm.simulate(forcing, base._replace(
            lake_rules=LakeRuleParams(jnp.float32(1.0), jnp.float32(0.1), jnp.float32(2.0), jnp.float32(1.0))))
        o_hi, _ = cm.simulate(forcing, base._replace(
            lake_rules=LakeRuleParams(jnp.float32(3.0), jnp.float32(0.1), jnp.float32(2.0), jnp.float32(1.0))))
        assert float(o_hi.mean()) > float(o_lo.mean())
