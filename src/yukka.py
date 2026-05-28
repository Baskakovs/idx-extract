"""Resolve Yukka entity IDs for assets via ISIN and RIC lookups."""

import logging
import os

import httpx
import polars as pl
from prefect import task
from prefect.artifacts import create_table_artifact

logger = logging.getLogger(__name__)

_BASE_URL = "https://metadata.api.yukkalab.com"
_BATCH_SIZE = 100


def _build_client() -> httpx.Client:
    """Build an authenticated HTTP client for the Yukka metadata API."""
    token = os.environ["YUKKA_TOKEN"]
    return httpx.Client(
        base_url=_BASE_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )


def _batch_lookup(client: httpx.Client, endpoint: str, identifiers: list[str]) -> dict[str, str]:
    """POST identifiers in batches and collect alpha_id mappings.

    Args:
        client: Authenticated HTTP client.
        endpoint: API endpoint path.
        identifiers: List of identifiers (ISINs or RICs) to look up.

    Returns:
        Mapping of identifier to alpha_id for successful lookups.
    """
    result: dict[str, str] = {}
    for i in range(0, len(identifiers), _BATCH_SIZE):
        batch = identifiers[i : i + _BATCH_SIZE]
        resp = client.post(endpoint, json=batch)
        resp.raise_for_status()
        data = resp.json()
        for key, entity in data.items():
            if entity is not None and "alpha_id" in entity:
                result[key] = entity["alpha_id"]
    return result


@task
def resolve_yukka_ids(assets_df: pl.DataFrame) -> pl.DataFrame:
    """Enrich assets DataFrame with yukka_id column via ISIN and RIC lookups.

    Args:
        assets_df: DataFrame with at least 'isin' and 'ric' columns.

    Returns:
        DataFrame with an added 'yukka_id' column (nullable string).
    """
    isins = assets_df["isin"].unique().to_list()
    logger.info("Resolving Yukka IDs for %d unique ISINs", len(isins))

    client = _build_client()
    try:
        # Primary: ISIN lookup
        isin_to_yukka = _batch_lookup(client, "/v2/isin_to_entity", isins)
        logger.info("ISIN lookup resolved %d / %d", len(isin_to_yukka), len(isins))

        # Fallback: RIC lookup for unresolved ISINs
        unresolved_isins = set(isins) - set(isin_to_yukka.keys())
        ric_to_yukka: dict[str, str] = {}
        if unresolved_isins:
            unresolved_df = assets_df.filter(pl.col("isin").is_in(list(unresolved_isins)))
            rics = unresolved_df["ric"].unique().drop_nulls().to_list()
            rics = [r for r in rics if r]
            if rics:
                ric_to_yukka = _batch_lookup(client, "/ric_to_entity", rics)
                logger.info("RIC lookup resolved %d / %d", len(ric_to_yukka), len(rics))
    finally:
        client.close()

    # Build ISIN -> yukka_id mapping combining both lookups
    isin_yukka_map: dict[str, str] = dict(isin_to_yukka)

    if ric_to_yukka:
        # Map RIC results back to ISINs
        ric_isin_df = assets_df.filter(pl.col("isin").is_in(list(unresolved_isins))).select("isin", "ric").unique()
        for row in ric_isin_df.iter_rows(named=True):
            if row["ric"] in ric_to_yukka and row["isin"] not in isin_yukka_map:
                isin_yukka_map[row["isin"]] = ric_to_yukka[row["ric"]]

    resolved_count = len(isin_yukka_map)
    logger.info("Total resolved: %d / %d ISINs", resolved_count, len(isins))

    # Build mapping DataFrame and join
    if isin_yukka_map:
        mapping_df = pl.DataFrame({"isin": list(isin_yukka_map.keys()), "yukka_id": list(isin_yukka_map.values())})
    else:
        mapping_df = pl.DataFrame({"isin": pl.Series([], dtype=pl.Utf8), "yukka_id": pl.Series([], dtype=pl.Utf8)})

    # Drop existing yukka_id column if present (re-enrichment)
    if "yukka_id" in assets_df.columns:
        assets_df = assets_df.drop("yukka_id")

    return assets_df.join(mapping_df, on="isin", how="left")


@task
def report_unresolved_assets(assets_df: pl.DataFrame) -> None:
    """Create a Prefect artifact reporting assets without a Yukka ID.

    Args:
        assets_df: Enriched DataFrame with 'yukka_id' column.
    """
    unresolved = assets_df.filter(pl.col("yukka_id").is_null())

    if len(unresolved) == 0:
        logger.info("All assets resolved to Yukka IDs")
        return

    logger.warning("%d assets could not be resolved to Yukka IDs", len(unresolved))

    report_cols = [
        c
        for c in ("isin", "ric", "name", "country", "currency", "first_included", "last_included")
        if c in unresolved.columns
    ]
    table = unresolved.select(report_cols).unique(subset=["isin"]).sort("isin").to_dicts()

    create_table_artifact(
        key="unresolved-yukka-assets",
        table=table,
        description=f"{len(table)} unique ISINs could not be resolved to Yukka entity IDs",
    )
