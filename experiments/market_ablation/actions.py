"""Archive-only corporate actions; these records never enter model features."""
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError
from datetime import date
from time import sleep
import json
import os
import pandas as pd
from .storage import atomic_json, read_json, event


def credentials():
    values = dict(os.environ)
    path = Path(__file__).resolve().parents[2] / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                values.setdefault(name.strip(), value.strip().strip("\"'"))
    key = values.get("APCA_API_KEY_ID", values.get("ALPACA_API_KEY"))
    secret = values.get("APCA_API_SECRET_KEY", values.get("ALPACA_API_KEY_SECRET"))
    if not key or not secret:
        raise RuntimeError("Alpaca credentials are unavailable for corporate-action accounting")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def fetch_actions(config):
    archive = json.loads((Path(config.archive)/"manifest.json").read_text())
    symbols = archive["request"]["symbols"]
    directory = config.root / "corporate_actions"
    complete = directory / "manifest.json"
    saved = read_json(complete)
    if saved and saved.get("complete"):
        return saved
    # Query process dates through today, then filter accounting by actual ex/effective date.
    records, pages = {}, []
    headers = credentials()
    for group in range(0, len(symbols), 50):
        token, index = None, 0
        while True:
            path = directory / "pages" / f"group-{group//50}-page-{index}.json"
            result = read_json(path)
            if result is None:
                query = {"symbols": ",".join(symbols[group:group+50]), "start": config.start,
                         "end": date.today().isoformat(), "limit": 1000, "sort": "asc", "data_quality": "all"}
                if token:
                    query["page_token"] = token
                request = Request("https://data.alpaca.markets/v1/corporate-actions?"+urlencode(query), headers=headers)
                for retry in range(6):
                    try:
                        with urlopen(request, timeout=60) as response:
                            result = json.load(response)
                        break
                    except HTTPError as error:
                        if error.code != 429 and error.code < 500:
                            raise RuntimeError(f"Corporate-action API returned HTTP {error.code}") from None
                        if retry == 5:
                            raise RuntimeError("Corporate-action API retries exhausted") from None
                        sleep(min(30, 2**retry))
                atomic_json(path, result)
            payload = result.get("corporate_actions", {})
            for kind, actions in payload.items():
                for action in actions:
                    identity = action.get("id", json.dumps(action, sort_keys=True))
                    records[identity] = {"type": kind, **action}
            pages.append(str(path.relative_to(directory)))
            token = result.get("next_page_token")
            if not token:
                break
            index += 1
        event(config, "corporate_actions", groups_complete=group//50+1, records=len(records))
    atomic_json(directory / "records.json", list(records.values()))
    summary = {"complete": True, "records": len(records), "pages": pages, "source": "Alpaca corporate-actions v1",
               "process_date_start": config.start, "process_date_end": date.today().isoformat(),
               "usage": "retrospective_economic_accounting_only", "input_price_adjustment": "raw"}
    atomic_json(complete, summary)
    fetch_recipient_prices(config, list(records.values()), headers)
    return summary


def fetch_recipient_prices(config, records, headers=None):
    symbols = json.loads((Path(config.archive)/"manifest.json").read_text())["request"]["symbols"]
    for record in records:
        if record["type"] != "spin_offs" or record.get("source_symbol") not in symbols:
            continue
        symbol, ex_date = record.get("new_symbol"), record.get("ex_date")
        if not symbol or not ex_date:
            continue
        path = config.root / "corporate_actions/recipient_prices" / f"{symbol}-{ex_date}.json"
        if path.exists():
            continue
        if headers is None:
            headers=credentials()
        begin = pd.Timestamp(ex_date, tz="UTC")
        query = {"symbols": symbol, "timeframe": "1Day", "start": begin.isoformat(),
                 "end": (begin+pd.Timedelta(days=10)).isoformat(), "feed": "sip", "adjustment": "raw", "limit": 1000}
        request = Request("https://data.alpaca.markets/v2/stocks/bars?"+urlencode(query), headers=headers)
        try:
            with urlopen(request, timeout=60) as response:
                data = json.load(response)
            atomic_json(path, {"symbol": symbol, "ex_date": ex_date, "source": "Alpaca raw SIP daily", "response": data})
        except HTTPError as error:
            atomic_json(path, {"symbol": symbol, "ex_date": ex_date, "status": "unavailable", "http_code": error.code})
        event(config, "recipient_prices", symbol=symbol, ex_date=ex_date)


if __name__ == "__main__":
    from .config import SweepConfig
    config = SweepConfig()
    fetch_actions(config)
    fetch_recipient_prices(config, read_json(config.root / "corporate_actions/records.json"))
