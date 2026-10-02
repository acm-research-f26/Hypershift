"""Reproducible current GICS metadata from public Vanguard holdings and MSCI.

Vanguard's JSON field named ``sector`` contains the sub-industry label shown in
its holdings table. Match that label exactly to MSCI's official taxonomy, then
derive the parent codes. Holdings dates do not establish historical information
availability: these records deliberately require ``retrospective_static``.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pandas as pd


MSCI_STRUCTURE_URL = "https://www.msci.com/documents/1296102/23c8ec04-fd1c-3518-e04c-4aa37027889d"
VANGUARD_URLS = {
    "VTI": "https://advisors.vanguard.com/investments/products/api/funds/0970/holdings/latest",
    "VT": "https://advisors.vanguard.com/investments/products/api/funds/3141/holdings/latest",
}
# Explicit issuer matches for foreign ordinary shares / US depositary receipts.
# Country + SEDOL + issuer name distinguish parents from unrelated tickers and
# separately listed subsidiaries. No company-name fuzzy matching is performed.
FOREIGN_ISSUERS = {
    "SAP": ("SAP", "DE", "4846288", "SAP SE"),
    "LOGI": ("LOGN", "CH", "B18ZRK2", "Logitech International SA"),
    "ENB": ("ENB", "CA", "2466149", "Enbridge Inc"),
    "TRP": ("TRP", "CA", "BJMY6G0", "TC Energy Corp"),
    "AZN": ("AZN", "GB", "0989529", "AstraZeneca PLC"),
    "NVS": ("NOVN", "CH", "7103065", "Novartis AG"),
    "SNY": ("SAN", "FR", "5671735", "Sanofi SA"),
    "GSK": ("GSK", "GB", "BN7SWP6", "GSK PLC"),
    "UL": ("ULVR", "GB", "BVZK7T9", "Unilever PLC"),
}
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _label(value):
    return " ".join(str(value).split()).casefold()


def _taxonomy_name(value):
    # The official sheet annotates changes and footnotes in the name cells.
    value = re.sub(r"\s*\((?:New\b|Sector Change\b|Definition Update\b)[^)]*\)\s*$", "", value, flags=re.I)
    return " ".join(value.replace("*", "").split())


def parse_msci_structure(payload):
    """Read only the visible current structure; exclude discontinued codes."""
    with ZipFile(BytesIO(payload)) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = [s for s in workbook.find("m:sheets", _NS)
                  if s.get("state", "visible") == "visible" and s.get("name", "").startswith("Effective close")]
        if len(sheets) != 1:
            raise ValueError("MSCI workbook must identify one visible current structure sheet")
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") for r in relationships if r.get("Id") == sheets[0].get(_REL))
        sheet_path = target.lstrip("/") if target.startswith("/") else "xl/" + target
        strings = ["".join(t.text or "" for t in s.iterfind(".//m:t", _NS))
                   for s in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        rows = []
        for row in ET.fromstring(archive.read(sheet_path)).findall("m:sheetData/m:row", _NS):
            cells = {}
            for cell in row:
                value = cell.find("m:v", _NS)
                if value is not None:
                    cells[re.sub(r"\d", "", cell.get("r"))] = strings[int(value.text)] if cell.get("t") == "s" else value.text
            rows.append(cells)
    if not any(all(row.get(c) == title for c, title in (("A", "Sector"), ("C", "Industry Group"),
                                                       ("E", "Industry"), ("G", "Sub-Industry"))) for row in rows):
        raise ValueError("Unrecognized MSCI structure columns")
    names, subindustries = {}, {}
    for row in rows:
        for column, width, name_column in (("A", 2, "B"), ("C", 4, "D"), ("E", 6, "F")):
            code, name = row.get(column, ""), row.get(name_column, "")
            if re.fullmatch(rf"\d{{{width}}}", code) and name and "discontinued" not in name.casefold():
                names[code] = _taxonomy_name(name)
        code, name = row.get("G", ""), row.get("H", "")
        if re.fullmatch(r"\d{8}", code) and name and "discontinued" not in name.casefold():
            name = _taxonomy_name(name)
            key = _label(name)
            if key in subindustries or code in {v[0] for v in subindustries.values()}:
                raise ValueError("Ambiguous active GICS sub-industry in MSCI structure")
            subindustries[key] = (code, name)
    taxonomy = {}
    for key, (code, name) in subindustries.items():
        if any(code[:width] not in names for width in (2, 4, 6)):
            raise ValueError(f"Missing GICS parent for {code}")
        taxonomy[key] = {"sector": code[:2], "sector_name": names[code[:2]],
                         "industry_group": code[:4], "industry_group_name": names[code[:4]],
                         "industry": code[:6], "industry_name": names[code[:6]],
                         "sub_industry": code, "sub_industry_name": name}
    if not taxonomy:
        raise ValueError("MSCI structure has no active classifications")
    return taxonomy, sheets[0].get("name")


def _ticker(value):
    return str(value).strip().upper().replace("/", ".").replace("-", ".").replace(" ", ".")


def classifications_from_holdings(node_ids, holdings, taxonomy, retrieved_at):
    """Require complete, unambiguous coverage; never infer unknown labels."""
    node_ids = tuple(node_ids)
    if not node_ids or len(set(node_ids)) != len(node_ids) or any(not isinstance(n, str) or not n.strip() for n in node_ids):
        raise ValueError("Require unique nonempty stock IDs")
    retrieval = pd.Timestamp(retrieved_at)
    if pd.isna(retrieval) or retrieval.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware")
    candidates = []
    for fund, payload in holdings.items():
        as_of = payload["latestEffectiveDate"]
        if pd.Timestamp(as_of).date() > retrieval.date():
            raise ValueError("Holdings reporting date is after retrieval")
        candidates.extend((fund, as_of, row) for row in payload[as_of]["equity"])
    records, missing = [], []
    for node in node_ids:
        if node in FOREIGN_ISSUERS:
            ticker, country, sedol, name = FOREIGN_ISSUERS[node]
            matches = [(fund, date, row) for fund, date, row in candidates
                       if row.get("ticker") == ticker and row.get("country") == country
                       and row.get("sedol") == sedol and row.get("holdingName") == name]
            identity_rule = "explicit_foreign_issuer"
        else:
            matches = [(fund, date, row) for fund, date, row in candidates
                       if _ticker(row.get("ticker", "")) == _ticker(node) and row.get("country") == "US"]
            identity_rule = "us_ticker"
        if not matches:
            missing.append(node)
            continue
        labels = {_label(row.get("sector", "")) for _, _, row in matches}
        identities = {row.get("sedol") or row.get("cusip") for _, _, row in matches}
        if len(labels) != 1 or len(identities) != 1 or None in identities or "" in identities:
            raise ValueError(f"Conflicting classifications or security identities for {node}")
        label = next(iter(labels))
        if label not in taxonomy:
            raise ValueError(f"Unrecognized GICS sub-industry for {node}: {label!r}")
        # Prefer the freshest reporting date, then VTI for US securities.
        fund, as_of, row = sorted(matches, key=lambda m: (m[1], m[0] == "VTI"))[-1]
        records.append({"node_id": node, **taxonomy[label],
                        "source": "Vanguard public holdings; codes from MSCI official GICS structure",
                        "source_url": VANGUARD_URLS[fund], "taxonomy_source_url": MSCI_STRUCTURE_URL,
                        "fund": fund, "holdings_as_of": as_of,
                        "retrieved_at": retrieval.tz_convert("UTC").isoformat(),
                        "metadata_protocol": "retrospective_static", "identity_rule": identity_rule,
                        "provider_ticker": row["ticker"], "provider_country": row["country"],
                        "provider_name": row["holdingName"], "cusip": row.get("cusip", ""),
                        "sedol": row.get("sedol", "")})
    if missing:
        raise ValueError("Public holdings do not cover stock IDs: " + ", ".join(missing))
    return pd.DataFrame(records)


def _download(url):
    request = Request(url, headers={"User-Agent": "Hypershift GICS research (public fund disclosures)"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def refresh_public_gics(universe_path, output_directory):
    """Fetch once into a new snapshot directory; experiment fitting is offline."""
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("Use a new GICS snapshot directory to preserve earlier experiments")
    universe_bytes = Path(universe_path).read_bytes()
    universe = pd.read_csv(BytesIO(universe_bytes), dtype=str)
    if "symbol" not in universe:
        raise ValueError("Universe CSV requires a symbol column")
    raw = {"msci-structure.xlsx": _download(MSCI_STRUCTURE_URL)}
    holdings = {}
    for fund, url in VANGUARD_URLS.items():
        raw[f"{fund}-holdings.json"] = _download(url)
        holdings[fund] = json.loads(raw[f"{fund}-holdings.json"])
    retrieved_at = datetime.now(timezone.utc).isoformat()
    taxonomy, sheet = parse_msci_structure(raw["msci-structure.xlsx"])
    records = classifications_from_holdings(universe.symbol, holdings, taxonomy, retrieved_at)
    raw["universe.csv"] = universe_bytes
    raw["classifications.csv"] = records.to_csv(index=False).encode("utf-8")
    raw["taxonomy.csv"] = pd.DataFrame(taxonomy.values()).sort_values("sub_industry").to_csv(index=False).encode("utf-8")
    manifest = {
        "metadata_protocol": "retrospective_static", "retrieved_at": retrieved_at,
        "taxonomy_sheet": sheet, "taxonomy_source_url": MSCI_STRUCTURE_URL,
        "provider_urls": VANGUARD_URLS, "holdings_as_of": {f: h["latestEffectiveDate"] for f, h in holdings.items()},
        "node_count": len(records), "missing_nodes": [],
        "foreign_issuer_matches": sorted(set(records.node_id) & FOREIGN_ISSUERS.keys()),
        "group_counts": {level: int(records[level].nunique()) for level in ("sector", "industry_group", "industry", "sub_industry")},
        "sha256": {name: sha256(payload).hexdigest() for name, payload in raw.items()},
        "scope": "Public holdings snapshot; classification effective dates and historical publication times are unknown."
                 " Do not report this as a point-in-time classification backtest.",
    }
    output.mkdir(parents=True)
    for name, payload in raw.items():
        (output / name).write_bytes(payload)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output / "classifications.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New snapshot directory")
    args = parser.parse_args()
    path = refresh_public_gics(args.universe, args.output)
    print(f"Saved {len(pd.read_csv(path))} authentic public GICS classifications to {path}")


if __name__ == "__main__":
    main()
