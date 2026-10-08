"""Publish current human-readable reports and their local assets into docs/."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

WORKSPACE = Path(__file__).resolve().parent
SOURCE = WORKSPACE / 'runs/robust_comparison'
DOCS = WORKSPACE / 'docs'


def main():
    DOCS.mkdir(exist_ok=True)
    mapping = {}
    for suffix in ['*.md', '*.csv']:
        for source in SOURCE.glob(suffix):
            mapping[source.resolve()] = DOCS / source.name
    for name in ['ROBUST_RESEARCH.md', 'STOCKMIXER_PAPER_REVIEW.md', 'ARCHITECTURE_RESEARCH.md']:
        mapping[WORKSPACE / name] = DOCS / name
    for source in (SOURCE / 'figures').glob('*.png'):
        mapping[source.resolve()] = DOCS / 'figures' / source.name
    for name in ['chronology_audit.json', 'confidence_interval_audit.json', 'combined_curve_audit.json', 'FINAL_VERIFICATION.json']:
        mapping[SOURCE / name] = DOCS / 'audit' / name
    # Remap Markdown paths so copied reports keep working from docs/.
    def rewritten(source, dest):
        def replace(match):
            prefix, target = match.group(1), match.group(2)
            if re.match(r'^[a-zA-Z]+://', target) or target.startswith('#'):
                return match.group(0)
            link, separator, fragment = target.partition('#')
            resolved = (source.parent / link.strip('<>')).resolve()
            mapped = mapping.get(resolved, resolved)
            relative = os.path.relpath(mapped, dest.parent).replace('\\', '/')
            return prefix + relative + (separator + fragment if separator else '') + ')'
        return re.sub(r'(!?\[[^\]]*\]\()([^)]+)\)', replace, source.read_text(encoding='utf-8-sig'))
    records = []
    for source, dest in mapping.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == '.md':
            dest.write_text(rewritten(source, dest), encoding='utf-8')
        else:
            shutil.copyfile(source, dest)
        records.append(dict(source=str(source.relative_to(WORKSPACE)).replace('\\','/'),
                            published=str(dest.relative_to(WORKSPACE)).replace('\\','/'),
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            published_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    deck = DOCS / 'slides/Hypershift_Pooled_Datasets_2022_2025.pptx'
    assert deck.exists(), 'Build and verify the updated presentation before publishing docs.'
    text = '''# Hypershift: current 2022–2025 study

Start here for the completed pooled study. Updated October 8, 2026.

| Document | Contents |
|---|---|
| [Dataset slides](slides/Hypershift_Pooled_Datasets_2022_2025.pptx) | Four editable slides with source links, chronological splits, pooled-return accounting and uncertainty |
| [Training dates and leakage audit](DATA_AND_SPLITS.md) | Exact training, validation and test dates, label maturity, checks and remaining data limitations |
| [Combined curves and metric tables](COMBINED_2022_2025.md) | All models and SPY/QQQ across 2022–2025, with explicit Sharpe confidence intervals |
| [Confidence intervals](CONFIDENCE_INTERVALS.md) | Model Sharpe, paired ETF differences, selected procedures and Huber-versus-MSE intervals |
| [Complete chart collection](CHARTS.md) | Combined and annual curves, ablations and diagnostic figures |
| [Full study report](REPORT.md) | All architecture comparisons and evaluation assumptions |
| [Key findings](FINDINGS.md) | What helped, what failed and the limits of the evidence |
| [ETF controls](ETF_CONTROLS.md) | Matching SPY/QQQ evaluation windows |
| [Research protocol](ROBUST_RESEARCH.md) | Training settings, portfolios, cost assumptions and reproduction |
| [StockMixer paper review](STOCKMIXER_PAPER_REVIEW.md) | What the paper establishes and how our implementations differ |

## Actual training years

| Test year | Training-signal years | Validation year |
|---|---|---|
| 2022 | 2019–2020 | 2021 |
| 2023 | 2019–2021 | 2022 |
| 2024 | 2019–2022 | 2023 |
| 2025 | 2020–2023 | 2024 |

Raw prices start in 2018. Warmup moves the first model-training signal to January 15, 2019.
Each test year uses its own prior-data model. The pooled curve combines those annual test returns,
with annual cash resets and omitted boundary sessions. NYSE and NASDAQ remain separate cohorts.

The saved-data chronology checks passed all 16 folds and the pipeline tests passed all 19 tests.
No future-observation leakage was detected in the audited code paths. Public vendor revisions,
missing delisted histories and prior test reuse remain material limitations.

## Data tables and reproduction

- [Complete combined metrics](combined_2022_2025_metrics.csv)
- [Dated combined returns and wealth curves](combined_2022_2025_curves.csv)
- [Pooled Sharpe intervals](pooled_confidence_intervals.csv)
- [Paired Huber-versus-MSE intervals](huber_mse_pooled_intervals.csv)
- [Exact fold audit](chronology_audit.csv)

The CSV tables and figures accompany the reports in this folder. Trained checkpoints and
original experiment records remain in `../runs/robust_2022_2025_final/` and
`../runs/daily_2022_2025_final/`. This documentation bundle copies current generated reports,
rewrites local links, and records source hashes in [the publication manifest](publication_manifest.json).
Refresh it with `python publish_research_docs.py` after regenerating reports.
The previous 2024–2025 protocol is retained as historical reference in
[ARCHITECTURE_RESEARCH.md](ARCHITECTURE_RESEARCH.md).
'''
    (DOCS / 'README.md').write_text(text, encoding='utf-8')
    broken = []
    for doc in DOCS.glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if re.match(r'^[a-zA-Z]+://', target) or target.startswith('#'):
                continue
            target = target.split('#')[0].strip('<>')
            if target and not (doc.parent / target).exists() and target != 'publication_manifest.json':
                broken.append((doc.name, target))
    assert not broken, broken
    (DOCS / 'publication_manifest.json').write_text(json.dumps(dict(
        updated_on='2026-10-08', files=records,
        slideshow_sha256=hashlib.sha256(deck.read_bytes()).hexdigest(),
        markdown_local_links_verified=True), indent=2))
    print(f'Published {len(records)} reports, data tables, figures and audit files. All local Markdown links verified.')


if __name__ == '__main__':
    main()
