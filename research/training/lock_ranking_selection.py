"""Watch validation-policy completion and lock alpha before test evaluation."""
import json
import time
from report_huber_ranking import RUN, FAMILY, YEARS, lock_choices, write_json


def main():
    paths=[RUN/f'{dataset}_{year}'/f'{kind}_policy.json'
           for dataset in ['NYSE_recent','NASDAQ_recent'] for year in YEARS for kind in FAMILY]
    deadline=time.monotonic()+600
    while not all(path.exists() for path in paths):
        if time.monotonic()>deadline:
            raise RuntimeError('Validation-policy completion wait exceeded 10 minutes.')
        time.sleep(1)
    policies=[json.loads(path.read_text()) for path in paths]
    lock_choices(policies)
    before_metrics=not (RUN/'all_metrics.csv').exists()
    write_json(RUN/'RANKING_LOCK_TIMING.json',dict(
        selection_created_before_all_test_metrics=before_metrics,
        validation_only=True, time_unix=time.time(),
        note='All family selections derive solely from saved validation policies; the report verifies this same lock.'))
    print(f'Ranking-weight selection locked from validation policies; before final test-metric table: {before_metrics}',flush=True)


if __name__=='__main__':
    main()
