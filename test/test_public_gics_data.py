"""Public GICS ingestion: taxonomy changes, issuer identity, and temporal scope."""

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from experiments.datasets import UNIVERSE_FILE
from experiments.gics import PUBLIC_GICS_DIRECTORY, PUBLIC_GICS_FILE, public_gics_pipeline
from hyperedges import ConstructionContext, ConstructionHistory, build_hyperedge_snapshot, fit_hyperedge_pipeline
from hyperedges.gics import load_classification_records, select_classifications_at_cutoff
from hyperedges.gics.public_data import classifications_from_holdings, parse_msci_structure


@pytest.fixture
def taxonomy():
    """Small workbook includes both old and new codes for the same label."""
    rows = [
        {"A": "Sector", "C": "Industry Group", "E": "Industry", "G": "Sub-Industry"},
        {"A": "20", "B": "Industrials", "C": "2020", "D": "Commercial & Professional Services",
         "E": "202020", "F": "Professional Services", "G": "20202030",
         "H": "Data Processing & Outsourced Services (Sector Change, New Code & Definition Update)"},
        {"A": "45", "B": "Information Technology", "C": "4510", "D": "Software & Services",
         "E": "451020", "F": "IT Services", "G": "45102020",
         "H": "Data Processing & Outsourced Services (Discontinued)"},
        {"E": "451030", "F": "Software", "G": "45103010", "H": "Application Software"},
        {"G": "45103020", "H": "Systems Software"},
    ]
    uri = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    strings, sheet = ET.Element(f"{{{uri}}}sst"), ET.Element(f"{{{uri}}}worksheet")
    data = ET.SubElement(sheet, f"{{{uri}}}sheetData")
    index = 0
    for i, row in enumerate(rows, 1):
        target = ET.SubElement(data, f"{{{uri}}}row", r=str(i))
        for column, value in row.items():
            ET.SubElement(ET.SubElement(strings, f"{{{uri}}}si"), f"{{{uri}}}t").text = value
            cell = ET.SubElement(target, f"{{{uri}}}c", r=f"{column}{i}", t="s")
            ET.SubElement(cell, f"{{{uri}}}v").text = str(index)
            index += 1
    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{uri}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Effective close of Mar 17 2023" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/sharedStrings.xml", ET.tostring(strings))
        archive.writestr("xl/worksheets/sheet1.xml", ET.tostring(sheet))
    return parse_msci_structure(payload.getvalue())[0]


@pytest.fixture
def holdings():
    def row(ticker, country, sedol, name, label):
        return {"ticker": ticker, "country": country, "sedol": sedol,
                "holdingName": name, "sector": label, "cusip": ""}
    return {"VT": {"latestEffectiveDate": "2026-08-31", "2026-08-31": {"equity": [
        row("MSFT", "US", "2588173", "Microsoft Corp", "Systems Software"),
        row("SAP", "DE", "4846288", "SAP SE", "Application Software"),
        row("SAP", "CA", "OTHER", "Saputo Inc", "Unrelated label"),
        row("BR", "US", "FIXTURE", "Broadridge", "Data Processing & Outsourced Services"),
    ]}}}


def test_taxonomy_excludes_discontinued_codes_and_preserves_new_parent(taxonomy):
    current = taxonomy["data processing & outsourced services"]
    assert current["sub_industry"] == "20202030"
    assert current["industry_group"] == "2020" and current["sector"] == "20"
    assert "45102020" not in {item["sub_industry"] for item in taxonomy.values()}


def test_explicit_foreign_identity_handles_ticker_collision(taxonomy, holdings):
    records = classifications_from_holdings(("MSFT", "SAP", "BR"), holdings, taxonomy, "2026-10-02T12:00:00Z")
    assert records.node_id.tolist() == ["MSFT", "SAP", "BR"]
    assert records.sub_industry.tolist() == ["45103020", "45103010", "20202030"]
    assert records.loc[1, "provider_country"] == "DE"
    assert records.loc[1, "identity_rule"] == "explicit_foreign_issuer"
    assert "effective_from" not in records
    with pytest.raises(ValueError, match="Point-in-time"):
        select_classifications_at_cutoff(records, records.node_id, "2024-01-01T00:00:00Z")
    selected = select_classifications_at_cutoff(records, records.node_id, "2024-01-01T00:00:00Z",
                                               metadata_protocol="retrospective_static")
    assert selected.loc["SAP", "industry_group"] == "4510"


@pytest.mark.parametrize("fault, message", [("missing", "do not cover"), ("label", "Unrecognized GICS"),
                                           ("conflict", "Conflicting"), ("identity", "do not cover")])
def test_unverified_data_fails_instead_of_filling_labels(fault, message, taxonomy, holdings):
    changed = deepcopy(holdings)
    rows = changed["VT"]["2026-08-31"]["equity"]
    nodes = ("MSFT", "SAP")
    if fault == "missing":
        nodes += ("UNKNOWN",)
    elif fault == "label":
        rows[0]["sector"] = "manual enterprise_software"
    elif fault == "conflict":
        rows.append({**rows[0], "sector": "Application Software"})
    else:
        rows[1]["sedol"] = "WRONG_ISSUER"
    with pytest.raises(ValueError, match=message):
        classifications_from_holdings(nodes, changed, taxonomy, "2026-10-02T12:00:00Z")


def test_fetched_local_dataset_builds_actual_gics_and_knn_pipeline():
    if not PUBLIC_GICS_FILE.is_file():
        pytest.skip("Ignored public GICS snapshot is not available; refresh explicitly")
    manifest = json.loads((PUBLIC_GICS_DIRECTORY / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert sha256((PUBLIC_GICS_DIRECTORY / name).read_bytes()).hexdigest() == digest
    universe = pd.read_csv(UNIVERSE_FILE, dtype=str)
    records = load_classification_records(PUBLIC_GICS_FILE)
    assert records.node_id.tolist() == universe.symbol.tolist()
    assert manifest["node_count"] == len(records) == 250
    assert records.industry_group.nunique() == 21
    for level, width in (("sector", 2), ("industry_group", 4), ("industry", 6)):
        assert records[level].eq(records.sub_industry.str[:width]).all()
    nodes = tuple(records.node_id)
    sessions = pd.date_range("2024-01-02", periods=5, freq="B")
    times = pd.DatetimeIndex([day.tz_localize("UTC") + pd.Timedelta(hours=15, minutes=15 * bar)
                              for day in sessions for bar in range(20)])
    cutoff = times[-1] + pd.Timedelta(minutes=2)
    returns = pd.DataFrame(np.random.default_rng(37).normal(size=(len(times), len(nodes))), index=times, columns=nodes)
    history = ConstructionHistory(returns, pd.DataFrame(True, index=times, columns=nodes),
                                  np.tile([False] + [True] * 19, len(sessions)),
                                  pd.DatetimeIndex(np.repeat(sessions, 20)), times + pd.Timedelta(minutes=1), cutoff)
    context = ConstructionContext(nodes, history, cutoff)
    snapshot = build_hyperedge_snapshot(fit_hyperedge_pipeline(context, public_gics_pipeline()))
    gics, knn = snapshot.families
    assert gics.incidence.shape == (250, 21)
    assert gics.incidence.any(axis=1).all() and gics.incidence.sum(axis=0).ge(2).all()
    assert gics.diagnostics["metadata_protocol"] == "retrospective_static"
    assert knn.incidence.shape == (250, 250)
