"""Static per-HRU / per-reach inputs for jFUSE: glacier fraction and
lake/reservoir attributes.

These are domain attributes (not forcing or calibrated parameters) that the
glacier module and the lake/reservoir routing node consume:

* :func:`load_glacier_fraction` reads the per-GRU glacier fraction SYMFLUENCE
  derives from the RGI glacier intersection, aligned to catchment/HRU order.
* :func:`classify_lakes_onto_network` stamps HydroLAKES storage-discharge
  attributes onto the river-network reaches a lake sits on.

Both degrade gracefully (return ``None`` / the unchanged network) when the
source data is absent, so a domain without glaciers or lakes is unaffected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np


def load_glacier_fraction(
    project_dir,
    domain_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    """Return per-GRU glacier fraction ``[n_gru]`` in catchment/HRU order.

    Prefers SYMFLUENCE's precomputed ``glacier_fraction`` column in
    ``data/attributes/climate/climate_statistics.csv`` (written by the
    model-ready store builder), falling back to summing the glacier domain-type
    fractions (``domType_2`` clean-accumulation, ``domType_3`` clean-ablation,
    ``domType_4`` debris; ``domType_1`` upland is excluded) in the
    ``catchment_with_domain_type`` intersection shapefile.

    Returns ``None`` if no glacier attribute is available (=> no glacier).
    """
    log = logger or logging.getLogger(__name__)
    project_dir = Path(project_dir)

    csv = project_dir / "data" / "attributes" / "climate" / "climate_statistics.csv"
    if csv.exists():
        try:
            import pandas as pd

            df = pd.read_csv(csv)
            if "glacier_fraction" in df.columns:
                frac = np.clip(df["glacier_fraction"].to_numpy(dtype="float32"), 0.0, 1.0)
                if float(frac.max()) > 0.0:
                    log.info(
                        "Loaded glacier fraction for %d GRUs (mean=%.3f, max=%.3f) from %s",
                        frac.size,
                        float(frac.mean()),
                        float(frac.max()),
                        csv,
                    )
                return frac
        except Exception:  # noqa: BLE001 — fall through to the shapefile source
            log.debug("Could not read glacier_fraction from %s", csv, exc_info=True)

    shp = (
        project_dir
        / "shapefiles"
        / "catchment_intersection"
        / "with_domain_type"
        / "catchment_with_domain_type.shp"
    )
    if shp.exists():
        try:
            import geopandas as gpd

            gdf = gpd.read_file(shp)
            cols = [c for c in gdf.columns if c.startswith("domType_") and c != "domType_1"]
            if cols:
                frac = np.clip(gdf[cols].sum(axis=1).to_numpy(dtype="float32"), 0.0, 1.0)
                log.info(
                    "Loaded glacier fraction for %d GRUs (mean=%.3f) from %s",
                    frac.size,
                    float(frac.mean()),
                    shp,
                )
                return frac
        except Exception:  # noqa: BLE001
            log.debug("Could not read glacier domain types from %s", shp, exc_info=True)

    log.debug("No glacier-fraction source found under %s; glacier disabled.", project_dir)
    return None


def glacier_fraction_by_gru_id(
    project_dir,
    domain_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """Per-GRU glacier fraction keyed by ``GRU_ID`` via an RGI ∩ river-basins
    overlay, robust to GRU-set/ordering differences between the forcing and any
    precomputed glacier-intersection shapefile.

    Spatially intersects the cached RGI glacier outlines
    (``data/attributes/glaciers/cache/RGI2000-v7.0-G-06_iceland.shp`` etc.) with
    the domain river basins, giving glacier area / GRU area per ``GRU_ID``. The
    result is cached to ``settings/JFUSE/glacier_fraction_by_gru.csv`` so the
    overlay runs once (not per parallel worker).

    Returns ``{int(GRU_ID): float frac}`` or ``None`` if inputs are missing.
    """
    log = logger or logging.getLogger(__name__)
    project_dir = Path(project_dir)
    cache_csv = project_dir / "settings" / "JFUSE" / "glacier_fraction_by_gru.csv"

    if cache_csv.exists():
        try:
            import pandas as pd

            df = pd.read_csv(cache_csv)
            return {int(r.gru_id): float(r.glacier_fraction) for r in df.itertuples()}
        except Exception:  # noqa: BLE001
            log.debug("Could not read glacier-fraction cache %s", cache_csv, exc_info=True)

    rgi_dir = project_dir / "data" / "attributes" / "glaciers" / "cache"
    rgi_shps = list(rgi_dir.glob("RGI*G-0*_*.shp")) if rgi_dir.exists() else []
    rb_dir = project_dir / "shapefiles" / "river_basins"
    rb_shps = (
        (
            list(rb_dir.glob("*riverBasins_semidistributed.shp"))
            or list(rb_dir.glob("*riverBasins*.shp"))
        )
        if rb_dir.exists()
        else []
    )
    if not rgi_shps or not rb_shps:
        log.debug(
            "RGI (%d) or river-basins (%d) shapefiles missing for overlay.",
            len(rgi_shps),
            len(rb_shps),
        )
        return None

    try:
        import geopandas as gpd
        import pandas as pd

        basins = gpd.read_file(rb_shps[0])
        if "GRU_ID" not in basins.columns:
            log.warning("river basins lack GRU_ID; cannot key glacier fraction.")
            return None
        glac = pd.concat([gpd.read_file(s) for s in rgi_shps], ignore_index=True)
        glac = gpd.GeoDataFrame(glac, geometry="geometry", crs=gpd.read_file(rgi_shps[0]).crs)

        # Project to an equal-area CRS for honest area ratios (Iceland Lambert).
        proj = "EPSG:3057"
        basins = basins.to_crs(proj)
        glac = glac.to_crs(proj)
        basins["_gru_area"] = basins.geometry.area

        log.info(
            "Computing glacier fraction: overlaying %d glaciers ∩ %d GRUs...",
            len(glac),
            len(basins),
        )
        inter = gpd.overlay(
            basins[["GRU_ID", "_gru_area", "geometry"]],
            glac[["geometry"]],
            how="intersection",
            keep_geom_type=True,
        )
        inter["_ice_area"] = inter.geometry.area
        ice_by_gru = inter.groupby("GRU_ID")["_ice_area"].sum()

        frac = (ice_by_gru / basins.set_index("GRU_ID")["_gru_area"]).fillna(0.0)
        frac = frac.clip(0.0, 1.0)
        out = {int(g): float(f) for g, f in frac.items()}
        # Ensure every GRU appears (0 where no glacier).
        for g in basins["GRU_ID"].astype(int):
            out.setdefault(int(g), 0.0)

        try:
            cache_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"gru_id": list(out.keys()), "glacier_fraction": list(out.values())}
            ).to_csv(cache_csv, index=False)
            log.info(
                "Cached glacier fraction for %d GRUs (mean=%.4f) -> %s",
                len(out),
                float(pd.Series(list(out.values())).mean()),
                cache_csv,
            )
        except Exception:  # noqa: BLE001
            log.debug("Could not write glacier-fraction cache", exc_info=True)
        return out
    except Exception:  # noqa: BLE001
        log.warning("Glacier overlay failed; glacier disabled.", exc_info=True)
        return None


def glacier_dtemp_by_gru_id(
    project_dir,
    domain_name: str,
    lapse_rate: float = 0.0065,
    logger: Optional[logging.Logger] = None,
):
    """Per-GRU glacier-surface temperature offset (<=0 K) keyed by ``GRU_ID``.

    jFUSE drives ice melt with the GRU-mean air temperature, but a glacier sits
    on the higher (colder) part of the GRU, so the valley-mean temperature melts
    ice far too fast. For each GRU this lapses temperature from the GRU-mean
    elevation to the area-weighted median glacier-surface elevation (RGI
    ``zmed_m``) of the glaciers it contains::

        dtemp = -lapse_rate * max(glacier_zmed - gru_mean_elev, 0)

    Uses the same RGI ∩ river-basins overlay as :func:`glacier_fraction_by_gru_id`
    with GRU-mean elevation from ``settings/FUSE/subcatchment_attributes.csv``.
    Cached to ``settings/JFUSE/glacier_dtemp_by_gru.csv``.

    Returns ``{int(GRU_ID): float dtemp}`` or ``None`` if inputs are missing.
    """
    log = logger or logging.getLogger(__name__)
    cache_csv = project_dir / "settings" / "JFUSE" / "glacier_dtemp_by_gru.csv"

    if cache_csv.exists():
        try:
            import pandas as pd

            df = pd.read_csv(cache_csv)
            return {int(r.gru_id): float(r.glacier_dtemp) for r in df.itertuples()}
        except Exception:  # noqa: BLE001
            log.debug("Could not read glacier-dtemp cache %s", cache_csv, exc_info=True)

    rgi_dir = project_dir / "data" / "attributes" / "glaciers" / "cache"
    rgi_shps = list(rgi_dir.glob("RGI*G-0*_*.shp")) if rgi_dir.exists() else []
    rb_dir = project_dir / "shapefiles" / "river_basins"
    rb_shps = (
        (
            list(rb_dir.glob("*riverBasins_semidistributed.shp"))
            or list(rb_dir.glob("*riverBasins*.shp"))
        )
        if rb_dir.exists()
        else []
    )
    attr_csv = project_dir / "settings" / "FUSE" / "subcatchment_attributes.csv"
    if not rgi_shps or not rb_shps or not attr_csv.exists():
        log.debug("RGI/river-basins/attributes missing for glacier-dtemp overlay.")
        return None

    try:
        import geopandas as gpd
        import pandas as pd

        basins = gpd.read_file(rb_shps[0])
        if "GRU_ID" not in basins.columns:
            log.warning("river basins lack GRU_ID; cannot key glacier dtemp.")
            return None
        glac = pd.concat([gpd.read_file(s) for s in rgi_shps], ignore_index=True)
        glac = gpd.GeoDataFrame(glac, geometry="geometry", crs=gpd.read_file(rgi_shps[0]).crs)
        if "zmed_m" not in glac.columns:
            log.warning("RGI glaciers lack zmed_m; cannot derive glacier dtemp.")
            return None

        proj = "EPSG:3057"  # Iceland Lambert, equal-area for honest area weights
        basins = basins.to_crs(proj)
        glac = glac.to_crs(proj)

        log.info(
            "Computing glacier dtemp: overlaying %d glaciers ∩ %d GRUs...",
            len(glac),
            len(basins),
        )
        inter = gpd.overlay(
            basins[["GRU_ID", "geometry"]],
            glac[["zmed_m", "geometry"]],
            how="intersection",
            keep_geom_type=True,
        )
        inter["_ice_area"] = inter.geometry.area
        # Area-weighted median glacier-surface elevation per GRU.
        num = inter.assign(_w=inter["zmed_m"] * inter["_ice_area"]).groupby("GRU_ID")["_w"].sum()
        den = inter.groupby("GRU_ID")["_ice_area"].sum()
        glac_zmed = (num / den).to_dict()

        gru_elev = pd.read_csv(attr_csv).set_index("gru_id")["elev_m"].to_dict()
        out = {}
        for gid, zmed in glac_zmed.items():
            ge = gru_elev.get(int(gid))
            if ge is None:
                continue
            out[int(gid)] = -float(lapse_rate) * max(float(zmed) - float(ge), 0.0)
        # Ensure every GRU appears (0 where no glacier).
        for g in basins["GRU_ID"].astype(int):
            out.setdefault(int(g), 0.0)

        try:
            cache_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"gru_id": list(out.keys()), "glacier_dtemp": list(out.values())}).to_csv(
                cache_csv, index=False
            )
            log.info(
                "Cached glacier dtemp for %d GRUs (min=%.2f K) -> %s",
                len(out),
                float(min(out.values())) if out else 0.0,
                cache_csv,
            )
        except Exception:  # noqa: BLE001
            log.debug("Could not write glacier-dtemp cache", exc_info=True)
        return out
    except Exception:  # noqa: BLE001
        log.warning("Glacier dtemp overlay failed; no glacier lapse applied.", exc_info=True)
        return None


def classify_lakes_onto_network(
    network_arrays,
    project_dir,
    domain_name: str,
    segid_field: str = "LINKNO",
    min_overlap_frac: float = 0.0,
    logger: Optional[logging.Logger] = None,
):
    """Stamp HydroLAKES storage-discharge attributes onto inline-lake reaches.

    Spatial-joins the HydroLAKES polygons
    (``data/attributes/lakes/domain_<name>_hydrolakes.gpkg``) against the
    river-network reach lines; each reach that intersects a lake is marked as a
    lake/reservoir node and its rating parameters are initialised from
    HydroLAKES (``Vol_total`` => storage capacity, ``Dis_avg`` => reference
    outflow, ``Lake_type`` => reservoir vs natural). When several lakes touch a
    reach the largest (by ``Lake_area``) wins.

    Args:
        network_arrays: The domain ``NetworkArrays`` (reach_ids are ``segid_field``).
        project_dir: ``domain_<name>`` project directory.
        domain_name: Domain name (for the HydroLAKES filename).
        segid_field: River-network segment-id field matching ``reach_ids``.
        min_overlap_frac: Minimum lake/reach intersection length fraction to
            treat a reach as a lake (0 => any intersection).
        logger: Optional logger.

    Returns:
        An updated ``NetworkArrays`` (lake fields set), or the original when no
        HydroLAKES data is present.
    """
    log = logger or logging.getLogger(__name__)
    project_dir = Path(project_dir)

    lakes_gpkg = (
        project_dir / "data" / "attributes" / "lakes" / f"domain_{domain_name}_hydrolakes.gpkg"
    )
    net_shp_dir = project_dir / "shapefiles" / "river_network"
    if not lakes_gpkg.exists():
        log.debug("No HydroLAKES file at %s; lakes disabled.", lakes_gpkg)
        return network_arrays
    net_shps = list(net_shp_dir.glob("*.shp")) if net_shp_dir.exists() else []
    if not net_shps:
        log.debug("No river-network shapefile under %s; cannot classify lakes.", net_shp_dir)
        return network_arrays

    try:
        import geopandas as gpd
        import jax.numpy as jnp

        lakes = gpd.read_file(lakes_gpkg)
        reaches = gpd.read_file(net_shps[0])
        if reaches.crs is not None and lakes.crs is not None and reaches.crs != lakes.crs:
            lakes = lakes.to_crs(reaches.crs)

        if segid_field not in reaches.columns:
            log.warning("segid field %r not in river network; cannot classify lakes.", segid_field)
            return network_arrays

        reach_ids = np.asarray(network_arrays.reach_ids)
        n = len(reach_ids)
        id_to_idx = {int(rid): i for i, rid in enumerate(reach_ids)}

        is_lake = np.zeros(n, dtype=bool)
        s_max = (
            np.array(network_arrays.lake_s_max)
            if network_arrays.lake_s_max is not None
            else np.zeros(n, np.float32)
        )
        q_ref = (
            np.array(network_arrays.lake_q_ref)
            if network_arrays.lake_q_ref is not None
            else np.zeros(n, np.float32)
        )
        q_min = (
            np.array(network_arrays.lake_q_min)
            if network_arrays.lake_q_min is not None
            else np.zeros(n, np.float32)
        )
        exp = (
            np.array(network_arrays.lake_exp)
            if network_arrays.lake_exp is not None
            else np.full(n, 2.0, np.float32)
        )
        spill = (
            np.array(network_arrays.lake_spill_coef)
            if network_arrays.lake_spill_coef is not None
            else np.full(n, 1.0, np.float32)
        )

        # Spatial join: lakes -> intersecting reaches.
        joined = gpd.sjoin(
            reaches[[segid_field, "geometry"]], lakes, predicate="intersects", how="inner"
        )
        n_set = 0
        # Keep the largest lake per reach.
        area_col = "Lake_area" if "Lake_area" in joined.columns else None
        if area_col:
            joined = joined.sort_values(area_col, ascending=False)
        seen = set()
        for _, row in joined.iterrows():
            rid = int(row[segid_field])
            if rid in seen or rid not in id_to_idx:
                continue
            seen.add(rid)
            i = id_to_idx[rid]
            is_lake[i] = True
            vol = float(row.get("Vol_total", 0.0) or 0.0) * 1.0e6  # mcm -> m³
            dis = float(row.get("Dis_avg", 0.0) or 0.0)  # m³/s
            ltype = int(row.get("Lake_type", 1) or 1)
            s_max[i] = max(vol, 1.0)
            q_ref[i] = max(dis, 0.0)
            # Reservoir (Lake_type 2/3) keeps a small minimum release; natural lake weir => 0.
            q_min[i] = 0.1 * max(dis, 0.0) if ltype != 1 else 0.0
            n_set += 1

        if n_set == 0:
            log.info("HydroLAKES present but no reaches intersect lakes; routing unchanged.")
            return network_arrays

        log.info(
            "Classified %d lake/reservoir reaches from HydroLAKES (%s).", n_set, lakes_gpkg.name
        )
        return network_arrays._replace(
            is_lake=jnp.asarray(is_lake),
            lake_s_max=jnp.asarray(s_max, dtype=jnp.float32),
            lake_q_ref=jnp.asarray(q_ref, dtype=jnp.float32),
            lake_q_min=jnp.asarray(q_min, dtype=jnp.float32),
            lake_exp=jnp.asarray(exp, dtype=jnp.float32),
            lake_spill_coef=jnp.asarray(spill, dtype=jnp.float32),
        )
    except Exception:  # noqa: BLE001 — never let lake classification break a run
        log.warning("Lake classification failed; routing without lakes.", exc_info=True)
        return network_arrays
