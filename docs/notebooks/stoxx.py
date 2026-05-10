# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.18.4",
#     "polars>=1.0",
#     "plotly>=5.18.0",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from datetime import date
    from pathlib import Path

    import marimo as mo
    import polars as pl

    # Resolve the project root from the notebook location
    project_root = Path(__file__).resolve().parents[2]

    # Ensure the src directory is importable regardless of how the notebook is launched
    _src = str(project_root / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)


@app.cell
def cell_02():
    """Render the notebook title and description."""
    mo.md(
        r"""
        # STOXX Index Explorer

        Interactive notebook for exploring STOXX index data stored as Parquet datasets.
        Use the controls below to browse indexes, review dates, and constituent details.
        """
    )
    return


@app.cell
def cell_03():
    """Create a text input for the dataset root path."""
    root_input = mo.ui.text(
        value=str(project_root / "data"),
        label="Dataset root path:",
        placeholder="Path to directory containing index datasets",
    )
    root_input
    return (root_input,)


@app.cell
def cell_04(root_input):
    """Load the ParquetRepository from the specified root path."""
    from repository import ParquetRepository
    from stoxx.repository import StoxxIndex

    root = Path(root_input.value)
    repo = ParquetRepository(root=root, index_factory=StoxxIndex)
    mo.stop(
        len(repo.indexes) == 0,
        mo.md(f"No index datasets found in `{root}`. Ensure the path contains subdirectories with `assets.parquet`."),
    )
    mo.md(f"Found **{len(repo.indexes)}** index(es): {', '.join(i.name for i in repo.indexes)}")
    return (repo,)


@app.cell
def cell_05(repo):
    """Create a dropdown to select an index."""
    index_dropdown = mo.ui.dropdown(
        options=[i.name for i in repo.indexes],
        value=repo.indexes[0].name,
        label="Select index:",
    )
    index_dropdown
    return (index_dropdown,)


@app.cell
def cell_06(index_dropdown, repo):
    """Show available review dates and create a dropdown to pick one."""
    selected_index = repo.get_index(index_dropdown.value)
    dates = selected_index.available_review_dates
    mo.stop(len(dates) == 0, mo.md("No review dates available for this index."))
    date_strings = [str(d) for d in dates]
    date_dropdown = mo.ui.dropdown(
        options=date_strings,
        value=date_strings[-1],
        label="Review date:",
    )
    mo.vstack(
        [
            mo.md(f"**{len(dates)}** review date(s) available"),
            date_dropdown,
        ]
    )
    return date_dropdown, selected_index


@app.cell
def cell_07():
    """Display the assets table."""
    mo.md("### Assets")
    return


@app.cell
def cell_07b(selected_index):
    """Show assets as an interactive table."""
    mo.ui.table(selected_index.assets.to_pandas())
    return


@app.cell
def cell_08(date_dropdown, selected_index):
    """Display constituents table for the selected review date."""
    _rd = date.fromisoformat(date_dropdown.value)
    _constituents = selected_index.constituents(_rd)
    mo.vstack(
        [
            mo.md(f"### Constituents ({date_dropdown.value}) &mdash; {len(_constituents)} members"),
            mo.ui.table(_constituents.to_pandas()),
        ]
    )
    return


@app.cell
def cell_09():
    """Explain the membership entry reason classes."""
    mo.callout(
        mo.md(
            r"""
            **Membership Entry Reasons**

            The STOXX Europe 600 uses a buffer rule to reduce turnover:

            | Reason | Description |
            |---|---|
            | `top_550` | Ranked 1-550 by free-float market cap — automatic entry |
            | `buffer_retained` | Ranked 551-750 and was already a member — retained via buffer rule |
            | `fill_to_600` | Fills remaining slots to reach 600, from largest non-members |
            | `bootstrap` | First review only — top 600 by market cap with no prior data |
            """
        ),
        kind="info",
    )
    return


@app.cell
def cell_10(date_dropdown, selected_index):
    """Entry reason breakdown bar chart."""
    import plotly.express as px

    _rd = date.fromisoformat(date_dropdown.value)
    _mem = selected_index.membership(_rd)
    _reason_counts = _mem.group_by("entry_reason").len().sort("len", descending=True).to_pandas()
    _fig = px.bar(
        _reason_counts,
        x="entry_reason",
        y="len",
        title=f"Entry Reason Breakdown ({date_dropdown.value})",
        labels={"entry_reason": "Entry Reason", "len": "Count"},
    )
    _fig.update_layout(template="plotly_white", height=400, showlegend=False)
    mo.ui.plotly(_fig)
    return (px,)


@app.cell
def cell_11(date_dropdown, selected_index):
    """Show entries with market cap and rank for the selected review date."""
    _rd = date.fromisoformat(date_dropdown.value)
    _entries = selected_index.entries(_rd)
    _has_mcap = "ff_mcap" in _entries.columns and _entries["ff_mcap"].null_count() < len(_entries)

    if _has_mcap:
        _ranked = _entries.filter(pl.col("rank").is_not_null()).sort("rank")
        mo.vstack(
            [
                mo.md(f"### Selection List Entries ({date_dropdown.value})"),
                mo.md(f"**{len(_ranked)}** ranked entries out of {len(_entries)} total"),
                mo.ui.table(_ranked.to_pandas()),
            ]
        )
    else:
        _ranked = _entries.filter(pl.col("rank").is_not_null()).sort("rank")
        mo.vstack(
            [
                mo.md(f"### Selection List Entries ({date_dropdown.value})"),
                mo.callout(mo.md("FF Market Cap data is not available for this review date."), kind="warn"),
                mo.md(f"**{len(_ranked)}** ranked entries out of {len(_entries)} total"),
                mo.ui.table(_ranked.to_pandas()),
            ]
        )
    return


@app.cell
def cell_12(date_dropdown, px, selected_index):
    """Market cap distribution of constituents."""
    _rd = date.fromisoformat(date_dropdown.value)
    _constituents = selected_index.constituents(_rd)

    if "ff_mcap" in _constituents.columns and _constituents["ff_mcap"].null_count() < len(_constituents):
        _mcap_data = _constituents.filter(pl.col("ff_mcap").is_not_null()).sort("ff_mcap", descending=True)
        _fig = px.histogram(
            _mcap_data.to_pandas(),
            x="ff_mcap",
            nbins=50,
            title=f"FF Market Cap Distribution ({date_dropdown.value})",
            labels={"ff_mcap": "FF Market Cap (MEUR)"},
        )
        _fig.update_layout(template="plotly_white", height=400, showlegend=False)
        _output = mo.vstack(
            [
                mo.md(f"### Market Cap Distribution ({date_dropdown.value})"),
                mo.md(f"- **Median:** {_mcap_data['ff_mcap'].median():,.0f} MEUR"),
                mo.md(f"- **Mean:** {_mcap_data['ff_mcap'].mean():,.0f} MEUR"),
                mo.md(f"- **Max:** {_mcap_data['ff_mcap'].max():,.0f} MEUR"),
                mo.ui.plotly(_fig),
            ]
        )
    else:
        _output = mo.callout(mo.md("FF Market Cap data is not available for this review date."), kind="warn")
    _output
    return


@app.cell
def cell_13(date_dropdown, selected_index):
    """Summary stats: member count, country distribution."""
    _rd = date.fromisoformat(date_dropdown.value)
    _constituents = selected_index.constituents(_rd)
    _country_counts = _constituents.group_by("country").len().sort("len", descending=True)
    _top_countries = _country_counts.head(10).to_pandas()

    mo.vstack(
        [
            mo.md(f"""### Summary ({date_dropdown.value})
    - **Total members:** {len(_constituents)}
    - **Countries represented:** {len(_country_counts)}
    """),
            mo.md("**Top 10 countries by member count:**"),
            mo.ui.table(_top_countries),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
