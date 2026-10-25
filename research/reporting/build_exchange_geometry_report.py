"""Create the Poppins PDF supplement and figures from completed diagnostics."""
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors

ROOT = Path('runs/hyperbolicity_20261010_exact')
OUT = Path('output/pdf/exchange_geometry_20261010')
OUT.mkdir(parents=True, exist_ok=True)
FONT = Path('presentations/build/canva_dataset/poppins/fonts')
for kind, suffix in [('Poppins', 'Regular'), ('PoppinsBold', 'Bold'), ('PoppinsBlack', 'Black')]:
    pdfmetrics.registerFont(TTFont(kind, str(FONT/f'Poppins-{suffix}.ttf')))
    font_manager.fontManager.addfont(str(FONT/f'Poppins-{suffix}.ttf'))
plt.rcParams.update({'font.family': 'Poppins', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
YELLOW, GREY, DARK = '#ffdc38', '#bcbcbc', '#101010'
s = pd.read_csv(ROOT/'geometry_summary.csv')
large = s[s.cohort.eq('large_company')]
matched = pd.read_csv(ROOT/'stock_count_sensitivity_summary.csv')
metrics = pd.read_csv('runs/largecap_20261010/comparison/combined_2022_2025_metrics.csv')
fixed = pd.read_csv('runs/largecap_20261010/comparison/fixed_k10_pooled.csv')
years = list(range(2022, 2026))
names = {'NYSE': 'NYSE_top100_20261010', 'NASDAQ': 'NASDAQ100_20261010'}


def values(frame, market, metric, column):
    return frame[frame.market.eq(market) & frame.metric.eq(metric)].sort_values('test_year')[column].to_numpy()


def dark_axes(ax):
    ax.set_facecolor(DARK)
    ax.tick_params(colors=GREY)
    ax.xaxis.label.set_color(GREY); ax.yaxis.label.set_color(GREY)
    for spine in ax.spines.values():
        spine.set_color('#464646')
    ax.grid(axis='y', color='#353535', linewidth=.6)


fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.0), facecolor=DARK)
for market, color in [('NYSE', GREY), ('NASDAQ', YELLOW)]:
    axes[0].plot(years, values(large, market, 'hypergraph_s1', 'largest_component_delta_hg'), marker='o', color=color, label=market, linewidth=2)
    axes[1].plot(years, values(large, market, 'model_inputs', 'delta_rel'), marker='o', color=color, label=market, linewidth=2)
for ax in axes:
    dark_axes(ax); ax.set_xticks(years); ax.set_xlabel('Year evaluated (geometry uses earlier training data)')
axes[0].set_ylim(0, 2.4); axes[0].set_yticks([0, .5, 1, 1.5, 2]); axes[0].set_ylabel('Network delta_hg (largest connected component)')
axes[1].set_ylim(.15, .29); axes[1].set_ylabel('Feature delta_rel (scaled model-input trajectories)')
axes[0].legend(frameon=False, labelcolor='white', loc='lower right')
axes[0].set_title('NETWORK SHAPE', color='white', fontweight='bold', pad=14)
axes[1].set_title('TIME-SERIES SHAPE', color='white', fontweight='bold', pad=14)
fig.suptitle('DOES GEOMETRY EXPLAIN THE GAP?', color='white', fontsize=22, fontweight='black', y=.98)
fig.text(.06, .895, 'Lower means more tree-like. Nasdaq has a modest feature difference; network delta is mostly tied.', color=GREY, fontsize=10)
fig.subplots_adjust(left=.07, right=.97, top=.80, bottom=.23, wspace=.28)
fig.text(.07, .08, 'Frozen large-company cohorts | Exact four-stock calculations | 2022-2025 folds\nFeature comparison: 95-97 NYSE and 83-86 Nasdaq stocks with complete training eligibility. Not an exchange-wide constant.', color=GREY, fontsize=8)
geometry_png = OUT/'exchange_geometry_comparison.png'
fig.savefig(geometry_png, dpi=180, facecolor=DARK); plt.close(fig)

display_models = [('think', 'THINK'), ('think_mix_huber', 'THINK + Mixer + Huber'), ('multiscale_huber', 'Multiscale + Huber'), ('momentum126', 'Simple momentum')]
fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=DARK)
positions = np.arange(len(display_models))
for offset, (market, color) in zip([-.18, .18], [('NYSE', GREY), ('NASDAQ', YELLOW)]):
    scores = [float(fixed[(fixed.horizon.eq(5)) & fixed.dataset.eq(names[market]) & fixed.model.eq(model)].net_sharpe.iloc[0]) for model, _ in display_models]
    bars = ax.bar(positions+offset, scores, width=.34, color=color, label=market, zorder=3)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x()+bar.get_width()/2, score+.035, f'{score:.2f}', color='white', ha='center', fontsize=10)
dark_axes(ax); ax.set_xticks(positions, [label for _, label in display_models], fontsize=9)
ax.set_ylim(0, 1.45); ax.set_ylabel('Net Sharpe (higher is better)')
ax.legend(frameon=False, labelcolor='white', loc='upper left')
ax.set_title('SAME TEN-STOCK PORTFOLIO RULE', color='white', fontweight='bold', fontsize=17, pad=20)
fig.text(.09, .02, 'Equal-weight top 10, five-session horizon, three-seed ensembles, costs included; pooled 2022-2025. Momentum has no hyperbolic network.', color=GREY, fontsize=8)
fig.subplots_adjust(left=.09, right=.97, top=.82, bottom=.18)
performance_png = OUT/'same_policy_model_comparison.png'
fig.savefig(performance_png, dpi=180, facecolor=DARK); plt.close(fig)

W, H = 595.28, 841.89
pdf_path = OUT/'Hypershift_Exchange_Hyperbolicity_Analysis.pdf'
c = canvas.Canvas(str(pdf_path), pagesize=(W, H))
c.setTitle('Hypershift - Does exchange geometry explain the performance gap?')
c.setAuthor('Hypershift research')
normal = ParagraphStyle('normal', fontName='Poppins', fontSize=9.1, leading=14, textColor=colors.HexColor(DARK))
small = ParagraphStyle('small', parent=normal, fontSize=7.7, leading=11.5)
bold = ParagraphStyle('bold', parent=normal, fontName='PoppinsBold', fontSize=11, leading=16)


def paragraph(text, y, style=normal):
    p = Paragraph(text, style)
    _, height = p.wrap(W-80, 1000)
    p.drawOn(c, 40, y-height)
    return y-height-9


def table(rows, y, widths, font_size=8):
    style = ParagraphStyle('cell', parent=small, fontSize=font_size, leading=font_size+4)
    cells = [[Paragraph(str(cell), style) for cell in row] for row in rows]
    t = Table(cells, colWidths=widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(YELLOW)),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7), ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    _, height = t.wrap(W-80, 1000); t.drawOn(c, 40, y-height)
    return y-height-13


def start(number, title, subtitle):
    c.setFillColor(colors.HexColor(DARK)); c.rect(0, H-117, W, 117, fill=1, stroke=0)
    c.setFillColor(colors.HexColor(YELLOW)); c.setFont('PoppinsBold', 8)
    c.drawString(40, H-30, 'HYPERSHIFT / RESEARCH SUPPLEMENT / 10 OCT 2026')
    c.setFillColor(colors.white); c.setFont('PoppinsBlack', 21); c.drawString(40, H-65, title)
    c.setFont('Poppins', 8.5); c.drawString(40, H-92, subtitle)
    c.setFillColor(colors.HexColor('#777777')); c.setFont('Poppins', 7)
    c.drawString(40, 24, 'Exact geometry diagnostics | Current-company retrospective cohorts | No new model fits')
    c.drawRightString(W-40, 24, f'{number} / 4')
    return H-138


y = start(1, 'DOES GEOMETRY EXPLAIN IT?', 'The result: some support in the features, limited support in the network.')
y = paragraph('<b>The hypothesis remains plausible, but these calculations do not confirm it.</b> Nasdaq has slightly lower relative feature delta in all four large-company training folds. The saved network delta is tied in three folds. Feature results also depend on scaling and which stocks are included.', y)
rows = [['Test year', 'NYSE network<br/>delta_hg*', 'Nasdaq network<br/>delta_hg*', 'NYSE feature<br/>delta_rel', 'Nasdaq feature<br/>delta_rel']]
for i, year in enumerate(years):
    rows.append([year, f"{values(large, 'NYSE', 'hypergraph_s1', 'largest_component_delta_hg')[i]:.1f}", f"{values(large, 'NASDAQ', 'hypergraph_s1', 'largest_component_delta_hg')[i]:.1f}", f"{values(large, 'NYSE', 'model_inputs', 'delta_rel')[i]:.4f}", f"{values(large, 'NASDAQ', 'model_inputs', 'delta_rel')[i]:.4f}"])
y = table(rows, y, [58, 114, 114, 115, 114])
y = paragraph('*Network results are for the largest connected component. The full saved graphs contain isolated stocks, so a single finite-distance global delta_hg is undefined. The components contain 99 NYSE stocks in every fold and 91, 92, 94 and 95 Nasdaq stocks respectively.', y, small)
c.drawImage(str(geometry_png), 40, y-241, width=W-80, height=241, mask='auto'); y -= 256
y = paragraph('In plain language: stocks that branch into related groups can be easier to represent in hyperbolic space. A lower delta means distances resemble a branching tree more closely. It does not measure forecasting skill, and there is no universal cutoff proving that a model should work.', y)
y = paragraph('The mean feature delta_rel is <b>0.2326 for NYSE</b> and <b>0.2182 for Nasdaq</b>, about 6.2% lower for Nasdaq. These are summaries of four overlapping training periods, not independent experiments or whole-exchange measurements.', y, small)
assert y > 40; c.showPage()

y = start(2, 'WHY NASDAQ MIGHT BE EASIER', 'Separate the performance gap from the effect of hyperbolic geometry.')
c.drawImage(str(performance_png), 40, y-225, width=W-80, height=225, mask='auto'); y -= 240
y = paragraph('A stronger candidate: different return patterns', y, bold)
y = paragraph('With the same equal-weight ten-stock rule, simple 126-session momentum reaches a net Sharpe of <b>1.17 on Nasdaq versus 0.83 on NYSE</b>. That rule has no hyperbolic network. This supports investigating trend persistence and the particular companies held, rather than attributing the entire gap to geometry. It is not a controlled causal test.', y)
y = paragraph('The advantage depends on the model and portfolio rule', y, bold)
rows = [['Five-session model', 'NYSE: policy<br/>chosen on validation', 'Nasdaq: policy<br/>chosen on validation']]
for model, label in display_models:
    scores = [metrics[metrics.horizon.eq(5) & metrics.dataset.eq(names[m]) & metrics.model.eq(model)].net_sharpe.iloc[0] for m in ['NYSE', 'NASDAQ']]
    rows.append([label, f'{scores[0]:.3f}', f'{scores[1]:.3f}'])
y = table(rows, y, [243, 136, 136])
y = paragraph('Plain THINK changes from a small Nasdaq advantage under fixed top-10 equal weights to a NYSE advantage under separately selected policies. The multiscale Huber model also favors Nasdaq, despite using Euclidean mixers rather than the THINK hyperbolic network. Better Nasdaq results therefore are not uniquely associated with hyperbolic architecture.', y, small)
y = paragraph('Sector mix, unusual winners and losses, tail events, and transaction costs are other plausible explanations. A tech/growth concentration story needs a matched sector or factor analysis. Our current training data do <b>not</b> show greater average co-movement on Nasdaq: mean pairwise return correlation is about 0.450 versus 0.474 on NYSE.', y, small)
assert y > 40; c.showPage()

y = start(3, 'HOW ROBUST IS THE DIFFERENCE?', 'Reasonable changes to the feature metric change the strength of the conclusion.')
y = paragraph('Large-company feature sensitivity', y, bold)
rows = [['Feature representation', 'NYSE mean<br/>delta_rel', 'Nasdaq mean<br/>delta_rel', 'Nasdaq lower<br/>folds']]
for metric, label in [('model_inputs', 'Actual scaled model inputs'), ('per_stock_standardized', 'Each stock standardized separately'), ('returns_only', 'Daily return channel only')]:
    ny = values(large, 'NYSE', metric, 'delta_rel'); na = values(large, 'NASDAQ', metric, 'delta_rel')
    rows.append([label, f'{ny.mean():.4f}', f'{na.mean():.4f}', f'{int((na<ny).sum())} / 4'])
y = table(rows, y, [251, 86, 90, 88])
y = paragraph('Equal-size check: 80 stocks per market', y, bold)
y = paragraph('For each fold and representation, I drew 40 pairs of 80-stock subsets and recalculated the exact maximum. For scaled model inputs, Nasdaq had lower delta_rel in <b>62.5%, 97.5%, 85.0% and 85.0%</b> of pairs, respectively. In 2022 the middle 80% of differences included both signs. These fractions describe sensitivity to stock inclusion; they are not statistical confidence levels.', y)
y = paragraph('The earlier samples tell a less stable story', y, bold)
old = s[s.cohort.eq('original_sample')]
rows = [['Earlier sample metric', 'NYSE mean', 'Nasdaq mean', 'Nasdaq lower<br/>folds']]
for metric, label in [('model_inputs', 'Scaled model-input delta_rel'), ('per_stock_standardized', 'Per-stock standardized delta_rel'), ('returns_only', 'Return-only delta_rel')]:
    ny = values(old, 'NYSE', metric, 'delta_rel'); na = values(old, 'NASDAQ', metric, 'delta_rel')
    rows.append([label, f'{ny.mean():.4f}', f'{na.mean():.4f}', f'{int((na<ny).sum())} / 4'])
y = table(rows, y, [251, 86, 90, 88])
y = paragraph('The earlier relationship networks have delta_hg = 1.0 in both markets in every fold, using their 68-stock NYSE and 67-stock Nasdaq connected components. Separate per-stock standardization reverses the earlier feature comparison in all four folds. This cautions against interpreting lower scaled-input delta as proof of a stable exchange-level hierarchy.', y, small)
y = paragraph('Additional checks', y, bold)
y = paragraph('Removing GOOGL while retaining GOOG leaves the primary Nasdaq feature delta_rel unchanged to four decimal places in every fold. Higher-overlap walks (s=2) are provided separately: they fragment the graphs substantially and are an alternative to the primary s=1 diagnostic. All feature and component results are exact over every four-stock subset; no sampled maximum is presented as a global exact value.', y, small)
assert y > 40; c.showPage()

y = start(4, 'WHAT WAS CALCULATED', 'Reproducible definitions, data boundaries and the next decisive experiment.')
y = paragraph('Definitions from the THINK paper', y, bold)
y = paragraph('The Gromov product is (x,y)_w = [d(w,x) + d(w,y) - d(x,y)] / 2. Delta is the smallest nonnegative value making (x,z)_w at least min[(x,y)_w, (y,z)_w] minus delta for all nodes.', y, small)
y = paragraph('An equivalent exact calculation uses every four stocks A, B, C, D:<br/>S1 = d(A,B) + d(C,D); S2 = d(A,C) + d(B,D); S3 = d(A,D) + d(B,C).<br/>Delta = maximum of (largest S - middle S) / 2.<br/><b>delta_hg:</b> use shortest s-walk network distances, with s=1 here.<br/><b>delta_rel:</b> use Euclidean distances between temporal stock vectors, then calculate 2 x delta / diameter. The diameter is the largest distance in that same feature metric.', y, small)
y = paragraph('Data and representation', y, bold)
y = paragraph('These diagnostics use the frozen current Nasdaq-100 and 100-largest-NYSE-company retraining cohorts. Hyperedges come directly from saved THINK checkpoints: each stock and up to four positively correlated neighbors, estimated from earlier training returns. The original paper instead uses industry and Wikidata relationships; our numbers do not reproduce its Table I.', y, small)
y = paragraph('For context, the paper reports NYSE delta_hg = 0.5 and delta_rel = 0.087, versus Nasdaq 1.0 and 0.107. Its own cohorts therefore have lower deltas on NYSE. They differ from ours in membership, period and representation, so those values cannot be transferred to this experiment.', y, small)
y = paragraph('One feature point is one stock. Its vector contains the chronological 16-session windows of five causal inputs at all five-session training origins. The saved training-fitted scaler, clipping at +/-8 standard deviations and 0.2 multiplier are reused. Overlapping windows retain their training-input weighting. The feature diameter and delta use the same vectors.', y, small)
y = paragraph('The primary feature comparison retains only stocks eligible at <b>every training origin</b>: 95, 95, 95, 97 NYSE names and 83, 83, 83, 86 Nasdaq names. This gives a finite, matched-calendar metric without inventing missing observations. It does <b>not</b> mean the remaining stocks lack usable model data: the forecasting model uses changing eligibility and admits later-listed companies as their histories accumulate.', y, small)
y = paragraph('Checks and limitations', y, bold)
y = paragraph('Saved splits, scaler values and checkpoint groups were matched to their reconstructed training inputs. The four-point calculation was independently checked against the paper\'s Gromov-product inequality, a tree, a four-node cycle, scale invariance and hyperedge overlap. Geometry sees no validation or test features. The 2026 membership lists still create retrospective selection bias. Windows overlap, and maxima are sensitive to extreme stocks and metric choices.', y, small)
y = paragraph('The experiment that could test the explanation', y, bold)
y = paragraph('Compare the same THINK architecture with hyperbolic versus Euclidean temporal and spatial layers, holding groups, inputs, seeds, optimization and portfolio rules fixed. Test whether the hyperbolic advantage grows as pre-test delta decreases across several matched universes and future periods. A Nasdaq-versus-NYSE comparison alone cannot isolate that effect. No additional forecasting models were trained for this diagnostic.', y, small)
y = paragraph('<b>Source:</b> Agarwal et al., THINK: Temporal Hypergraph Hyperbolic Network, ICDM 2022, equations (1)-(2), Appendix A equations (18)-(19), and Algorithm 1.<br/><link href="https://tylersnetwork.github.io/papers/icdm22-think.pdf" color="#333333">https://tylersnetwork.github.io/papers/icdm22-think.pdf</link><br/><b>Reproduction:</b> analyze_exchange_hyperbolicity.py and check_exchange_geometry_sensitivity.py; outputs in runs/hyperbolicity_20261010_exact. Performance comparisons reuse the completed large-company results.', y, small)
assert y > 40; c.showPage(); c.save()

main = large[large.metric.isin(['hypergraph_s1', 'model_inputs'])].copy()
main.to_csv(OUT/'exchange_geometry_results.csv', index=False)
packet = OUT/'Exchange_Geometry_Calculations.zip'
with zipfile.ZipFile(packet, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.iterdir():
        if p.is_file():
            z.write(p, f'calculations/{p.name}')
    for p in [Path('analyze_exchange_hyperbolicity.py'), Path('check_exchange_geometry_sensitivity.py'), Path('analyze_exchange_common_movement.py'), Path('archive/legacy/hyperbolicity_audit.py'), Path(__file__)]:
        z.write(p, f'code/{p.name}')
    for p in [Path('runs/largecap_20261010/comparison/combined_2022_2025_metrics.csv'), Path('runs/largecap_20261010/comparison/fixed_k10_pooled.csv')]:
        z.write(p, f'performance/{p.name}')
save = {'pdf': str(pdf_path.resolve()), 'figures': [str(geometry_png.resolve()), str(performance_png.resolve())],
        'data_packet': str(packet.resolve()), 'pages': 4, 'geometry_windows': 16, 'feature_metrics': 48,
        'equal_count_checks': 960, 'all_primary_calculations_exact': bool(s.exact.all())}
(OUT/'artifact_manifest.json').write_text(json.dumps(save, indent=2), encoding='utf-8')
print(json.dumps(save, indent=2))
