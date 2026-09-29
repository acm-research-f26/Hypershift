"""Generate an audit with formulas, bounds, witnesses and explicit missing inputs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path('runs/think_audit')


def read(name):
    return json.loads((ROOT / (name+'.json')).read_text())


def graph_witness(result):
    return max(result['components'], key=lambda r:r['lower_bound'])


def show_witness(result, heading):
    w = result['witness']
    if not w:
        return [f'### {heading}', '', 'All pair sums agree; delta is zero.', '']
    labels = w.get('labels',w.get('global_vertices',w['vertices']))
    distances = w['distances']
    s = w['pair_sums']
    ordered = sorted(s)
    lines = [f'### {heading}', '',
             'Assign A, B, C and D respectively to: **' + ', '.join(map(str,labels)) + '**.', '',
             '| Distance | Value |', '|---|---:|']
    lines += [f'| d({pair[0].upper()},{pair[1].upper()}) | {value:.9f} |' for pair,value in distances.items()]
    lines += ['', f'- S1 = d(A,B)+d(C,D) = {s[0]:.9f}',
              f'- S2 = d(A,C)+d(B,D) = {s[1]:.9f}',
              f'- S3 = d(A,D)+d(B,C) = {s[2]:.9f}',
              f'- Quadruple contribution = ({ordered[-1]:.9f} - {ordered[-2]:.9f}) / 2 = **{w["delta"]:.9f}**.', '',
              f"Diameter of this metric/component: {result['diameter']:.9f}. "
              f"Normalized witness value: 2 × {w['delta']:.9f} / {result['diameter']:.9f} = **{result['relative_lower_bound']:.9f}**.", '',
              f"Method: {result['method']}. Evaluated {result['quadruples_evaluated']:,} quadruples/draws out of "
              f"{result['possible_quadruples']:,} possible distinct unordered quadruples.", '']
    if result['exact']:
        lines += ['The calculation certifies this as the maximum for this specified metric, to floating-point precision.', '']
    else:
        lines += [f"This is a **lower bound**, not an exact maximum: delta lies in [{result['lower_bound']:.9f}, {result['upper_bound']:.9f}]. "
                  'The upper bound uses diameter/2. Sampling uses seed 20260928, with replacement across draws and four distinct vertices within a draw.', '']
    return lines


def main():
    cp = read('CPox_source_graph'); cf = read('CPox_raw_trajectories')
    wf = read('WMill_raw_trajectories')
    ny = read('NYSE_relations_s1'); nf = read('NYSE_RSR_feature_trajectories')
    na = read('NASDAQ_relations_s1'); af = read('NASDAQ_RSR_feature_trajectories')
    rg = read('DTT_rg17_neighborhood_s1'); uo = read('DTT_uo17_neighborhood_s1')
    lines = ['# Independent THINK hyperbolicity audit', '',
             '**Conclusion: the published table is not fully independently verified.** '
             'I retrieved original public source data and performed independent calculations, including exact small/medium metric evaluations '
             'and inspectable lower bounds for larger metrics. But the released material does not identify the exact processed '
             'incidence matrices and feature tensors behind the reported numbers. TSE and CSE remain uncomputed because I could '
             'not obtain the matching complete inputs. Differences below are conditional on declared reconstructions; they are not proof '
             'that the authors calculated their own inputs incorrectly.', '',
             '## Reported values and audit status', '',
             'The reported columns below come from Table I of [THINK](https://tylersnetwork.github.io/papers/icdm22-think.pdf). '
             'Graph delta and feature relative delta are different quantities. Our feature comparisons use one full temporal trajectory '
             'per node; that representation is an explicit assumption, not a recovered THINK setting.', '',
             '| Dataset | Reported graph delta | Reported feature relative delta | Independent result on declared source reconstruction |',
             '|---|---:|---:|---|',
             f'| CPox | 1.5 | 0.190 | Original county graph: exact 1.5. Unmerged neighborhood hypergraph: exact 1.0. Published FX trajectory relative delta: exact {cf["relative_lower_bound"]:.6f}. |',
             f'| WMill | 1.0 | 0.025 | All positive source connections give a complete graph: exact 0. Published trajectory relative delta: exact {wf["relative_lower_bound"]:.6f}. |',
             f'| NYSE | 0.5 | 0.087 | Reconstructed s=1 hypergraph has {ny["component_count"]} components; no finite global metric. Component delta bounds [{ny["max_component_lower_bound"]}, {ny["max_component_upper_bound"]}]. Feature relative delta at least {nf["relative_lower_bound"]:.6f}. |',
             f'| NASDAQ | 1.0 | 0.107 | Reconstructed s=1 hypergraph has {na["component_count"]} components; no finite global metric. Component delta bounds [{na["max_component_lower_bound"]}, {na["max_component_upper_bound"]}]. Feature relative delta at least {af["relative_lower_bound"]:.6f}. |',
             f'| DTT | 1.0 | Not reported | All 120 snapshots inspected for each public RG17/UO17 variant. Maximum component-delta bounds: RG17 [{rg["max_component_lower_bound"]}, {rg["max_component_upper_bound"]}]; UO17 [{uo["max_component_lower_bound"]}, {uo["max_component_upper_bound"]}]. No global disconnected-graph delta substituted. |',
             '| TSE | 1.5 | 0.074 | Not computed: matching 95-stock, 1,159-step panel and incidence matrix unavailable. |',
             '| CSE | 1.5 | 0.176 | Not computed: matching 85-stock, 1,293-step panel and incidence matrix unavailable. |', '',
             'A sampled maximum is always displayed as a lower bound unless it reaches a proven upper bound. '
             'The stock lower bounds already exceed the reported values under our feature representation, but **the representations '
             'have not been shown to be the same**. That prevents a valid claim that the paper is numerically wrong.', '',
             '## Calculation method, without a forecasting model', '',
             'No GNN is trained or used in these calculations. Hyperbolicity is calculated directly from a chosen distance matrix.', '',
             'In plain language, a hyperedge is a group, such as several companies in one industry. Hyperbolicity asks how '
             'the resulting distances compare with distances in a branching tree. It does not measure forecasting accuracy. '
             'A low value alone does not establish a useful financial hierarchy: even a complete graph has zero hyperbolicity '
             'when considering only its vertices. Whether a hyperbolic model helps must still be tested on future held-out outcomes '
             'against Euclidean hypergraphs and models with no graph.', '',
             'For graph geometry, H records which nodes belong to which hyperedges. The number of shared groups for nodes i,j is '
             'the (i,j) entry of H H-transpose. We connect distinct nodes if that count is at least s. Breadth-first search then gives '
             'the number of steps in the shortest unweighted path. Components are handled separately; infinity is never replaced by zero.', '',
             'For feature geometry, each node is represented by a declared vector. We calculate every Euclidean distance using '
             'd(i,j) = sqrt(sum_k (x[i,k]-x[j,k])^2). The implementation uses a Gram matrix to avoid allocating a huge node-by-node-by-feature tensor.', '',
             'For every four distinct nodes A,B,C,D, form S1=d(A,B)+d(C,D), S2=d(A,C)+d(B,D), and S3=d(A,D)+d(B,C). '
             'Sort the three sums. Half the gap between the largest and middle sum is that quadruple\'s contribution. '
             'Delta is the largest contribution over all quadruples. Relative delta is 2*delta/diameter, with diameter the largest '
             'pairwise distance in the metric. See the independent [SageMath definition](https://doc.sagemath.org/html/en/reference/graphs/sage/graphs/hyperbolicity.html).', '',
             'For at most 350 points we enumerate every distinct quadruple exactly once. Above that size we use 2,000,000 reproducible '
             'random draws, recording the best actual witness and the general diameter/2 upper bound. Hitting that upper bound certifies '
             'the maximum. Otherwise the result remains a bound. Exactness here concerns the supplied finite metric, not identity with '
             'the paper\'s missing processed dataset.', '',
             'The tests independently evaluate the Gromov-product inequality over all ordered quadruples of an eight-point example '
             'and confirm the same result as the four-point method. Additional checks cover a path, a clique, a four-cycle, disconnected '
             'graphs, scale invariance and the distinction between bounds and exact values.', '',
             '## CPox: what matches, and what does not', '',
             'The [public source](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/ea40a6a396b6688a94d7482d9d5fd288eaa2cb3b/dataset/chickenpox.json) '
             'has 20 counties and **521** FX observations, versus 522 time steps in the reported table. FX is already standardized: '
             'per-county means are approximately zero and standard deviations approximately one. It must not be described as unprocessed '
             'case counts. The artifact filename raw_trajectories means the published FX values without further changes.', '',
             'I symmetrized the provided county adjacency and removed self-loops. That graph gives 1.5. '
             'Its agreement with the table is a **numeric match on the original graph**, not verification of the merged hypergraph construction.', '']
    lines += show_witness(graph_witness(cp),'County graph: exhaustive witness')
    lines += ['![County graph and the four-point calculation](runs/think_audit/chickenpox_calculation.png)', '']
    lines += ['### Hyperedge-construction sensitivity', '', '| Construction | Exact delta |', '|---|---:|']
    for key,label in [('CPox_source_graph','Original county graph'),('CPox_neighborhood_s1','Closed neighborhoods, no merging, s=1'),
                      ('CPox_neighborhood_s2','Closed neighborhoods, no merging, s=2'),('CPox_dice_0.5','Greedy union of most similar pairs with Dice >=0.5'),
                      ('CPox_dice_0.75','Greedy union of most similar pairs with Dice >=0.75'),('CPox_literal_low_dice_0.5','Literal low-similarity interpretation: merge Dice <0.5')]:
        lines.append(f'| {label} | {read(key)["global_delta"]} |')
    lines += ['', 'These variants expose sensitivity to missing settings. Thresholds were declared as reconstruction examples, '
              'not tuned until a match appeared. Similarity, merging direction, union rule, tie order and s must all be fixed to replicate a result.', '']
    lines += show_witness(cf,'Published FX trajectories: exhaustive feature calculation')
    lines += ['Re-standardizing the published FX trajectories gives the same answer to floating-point precision. '
              'The difference from 0.190 is not explained by merely applying that standardization again. Different lag windows, '
              'point definitions or subsampling could still explain it.', '',
              '## WMill: the supplied adjacency is dense', '',
              'The [official loader](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/ea40a6a396b6688a94d7482d9d5fd288eaa2cb3b/torch_geometric_temporal/dataset/windmilllarge.py) '
              'links the downloaded 17,472-by-319 series. Its edge list has 101,761 entries, exactly 319 squared; every supplied weight '
              'is positive. Keeping every positive connection yields a complete graph. Every closed neighborhood then contains every node.', '',
              'For any four distinct nodes, all six unweighted distances are 1. Thus S1=S2=S3=2 and delta=(2-2)/2=**0**. '
              'Merging identical universal neighborhoods does not change the s=1 result. The paper\'s value 1.0 therefore requires '
              'additional choices beyond this literal positive-edge construction.', '',
              'As a sensitivity check, keeping only weights greater than 0.5 before creating closed neighborhoods yields four '
              'components with sizes 289, 26, 3 and 1; their maximum delta is exactly 2.0. This threshold is our declared example, '
              'not an asserted paper setting.', '']
    lines += show_witness(wf,'Published hourly trajectories: exhaustive feature calculation')
    ws = read('WMill_standardized_trajectories')
    lines += [f'Standardizing each turbine trajectory separately changes relative delta to **{ws["relative_lower_bound"]:.6f}**. '
              'Neither full-trajectory variant reproduces 0.025. That demonstrates the importance of defining temporal features precisely.', '']
    for market,graph,features in [('NYSE',ny,nf),('NASDAQ',na,af)]:
        shape = read(market+'_panel_shape')
        alternate = read(market+'_standardized_trajectories')
        lines += [f'## {market}: source universe recovered; processed hypergraph not recovered', '',
                  f"The downloaded panel has {shape['nodes']:,} stocks, {shape['timesteps']:,} time steps and five features per time. "
                  'Counts match the reported universe. Sources are the [original RSR repository](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking) '
                  'linked by the [STHAN-SR release](https://github.com/midas-research/sthan-sr-aaai).', '',
                  'Our reconstructed groups use the supplied named industries, excluding the missing-industry category n/a. '
                  'For the supplied selected Wikidata paths, first-order links form source-plus-target groups by relation; '
                  'second-order links form pairs. Identical memberships are deduplicated. These are inspectable assumptions; '
                  'the exact THINK relation snapshot, filtering and duplicate policy remain unknown.', '',
                  f"At s=1 there are {graph['component_count']} components, with the largest containing {graph['component_sizes'][0]} nodes. "
                  'Consequently there is no finite global shortest-path metric. The following is a witness inside a component, '
                  'not a global-delta replacement.', '']
        lines += show_witness(graph_witness(graph),'Reconstructed relationship metric: witness and bound')
        s2 = read(market+'_relations_s2')
        lines += [f"At s=2, the same memberships instead yield {s2['component_count']} components, with exact maximum component "
                  f"delta {s2['max_component_lower_bound']}. This is another sensitivity calculation.", '',
                  'For feature geometry, I follow the cited RSR loader\'s five-feature inputs: NASDAQ drops the final source row; '
                  f"sentinel -1234 values are replaced by 1.1 ({shape['sentinel_values_replaced']:,} replacements here). "
                  'I then flatten each stock\'s full feature history into one point. This deliberately reproduces that published '
                  'loader convention, but does not establish that THINK used these exact feature vectors for its geometry table.', '']
        lines += show_witness(features,'Full processed feature trajectories: sampled lower bound')
        lines += [f"After separately standardizing each per-stock feature trajectory, the relative-delta lower bound becomes "
                  f"{alternate['relative_lower_bound']:.6f}. This alternative is also recorded, not selected as the supposed original.", '']
    lines += ['## DTT: dynamic and disconnected', '',
              'The public PyTorch Geometric Temporal release contains both [Roland-Garros RG17](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/ea40a6a396b6688a94d7482d9d5fd288eaa2cb3b/dataset/twitter_tennis_rg17.json) '
              'and [US Open UO17](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/ea40a6a396b6688a94d7482d9d5fd288eaa2cb3b/dataset/twitter_tennis_uo17.json). '
              'I inspected all 120 timestamps in each variant, retaining the full 1,000-node universe, symmetrizing mentions '
              'and creating unmerged closed-neighborhood hyperedges with s=1. No feature relative-delta number is reported for DTT in THINK.', '']
    for event,data in [('RG17',rg),('UO17',uo)]:
        best_snapshot = max(data['snapshots'],key=lambda r:r['max_component_lower_bound'])
        best_component = graph_witness(best_snapshot)
        lines += [f"{event}: {data['disconnected_snapshots']}/{data['snapshot_count']} snapshots are disconnected. "
                  f"The maximum component delta across snapshots lies in [{data['max_component_lower_bound']}, {data['max_component_upper_bound']}]. "
                  f"All component calculations exact: {data['all_components_exact']}. "
                  'This time aggregation is our explicit audit summary, not an identified aggregation from the paper.', '']
        lines += show_witness(best_component,f'{event}: largest found witness, snapshot {best_snapshot["snapshot"]}')
    lines += ['Per-snapshot component counts, bounds and witnesses are saved in the DTT JSON files. '
              'Components smaller than four nodes have zero vertex-metric delta and are summarized by their sizes.', '',
              '## TSE: not computed', '',
              'The [cited source repository](https://github.com/liweitj47/overnight-stock-movement-prediction) links a '
              'Baidu archive described as original news data. That link could not be retrieved through the available browsing interface. '
              'Its repository does not supply the matching 95-stock panel and THINK incidence matrix. The STHAN-SR release contains '
              'TSE loading code but no complete matching files; its [TSE dataset issue](https://github.com/midas-research/sthan-sr-aaai/issues/11) '
              'also provides no resolution in the inspected page. No witness or numerical estimate is invented for this dataset.', '',
              '## CSE: not computed', '',
              'The [cited source paper](https://arxiv.org/pdf/1805.07979) links historical Baidu archives for news and social data; '
              'those links could not be retrieved through the browsing interface. Following related authors\' releases through '
              '[HYPHEN](https://github.com/gtfintechlab/HYPHEN-ACL) to [FAST](https://github.com/midas-research/fast-eacl) identifies '
              'related data and sample files, but not a verified complete 85-node, 1,293-step THINK panel with incidence matrix. '
              'The cited source discusses a different original sampling period/universe, so the precise filtering/history behind '
              'THINK remains unresolved. No surrogate current-market download is labeled as the original CSE dataset.', '',
              '## What would permit a conclusive verification?', '',
              'The [THINK repository](https://github.com/shivamag125/ICDM22-THINK) returned GitHub API status 409, '
              '"Git Repository is empty," during this audit. To independently check the exact table, we need:', '',
              '1. The incidence matrix for each static dataset, and each dynamic snapshot, with node order and file hashes.',
              '2. The actual temporal feature tensor used for relative delta: what constitutes a point, lag length, date range and scaling.',
              '3. The s value, edge direction/weight handling, duplicate policy and treatment of disconnected nodes.',
              '4. The neighborhood-merging threshold, merge direction and update/tie rules.',
              '5. The exact-versus-sampled calculation procedure, including seeds/sample sizes and DTT time aggregation.', '',
              'Without these, I can verify the mathematics on explicit inputs and identify unresolved discrepancies, '
              'but cannot responsibly certify or reject the entire published table. A numerical match produced by trying '
              'many undocumented settings would not resolve this problem.', '',
              '## Reproduce and inspect', '',
              '```powershell',
              '.\\.venv\\Scripts\\python.exe fetch_think_sources.py',
              '.\\.venv\\Scripts\\python.exe fetch_stock_audit.py',
              '.\\.venv\\Scripts\\python.exe -m unittest test_audit -v',
              '.\\.venv\\Scripts\\python.exe run_think_audit.py --section public',
              '.\\.venv\\Scripts\\python.exe run_think_audit.py --section stocks',
              '.\\.venv\\Scripts\\python.exe report_think_audit.py',
              '```', '',
              'The public calculation can take several minutes because it includes exhaustive evaluations and all Twitter snapshots. '
              'Large stock metrics return declared bounds. Downloads use commit-pinned sources where available and retain their hashes. '
              'The compressed relation archive expands to several GB if fully extracted; our script reads only its small JSON/text members, '
              'not its huge dense NumPy arrays.', '',
              'A separate pre-publication revision check confirms that the downloaded Chickenpox and both Twitter files are '
              'byte-for-byte identical to their latest GitHub revisions before October 2022. The old Windmill loader pointed to '
              'graphmining.ai, which failed DNS resolution during this audit; we used the current official loader\'s Box mirror '
              'and cannot prove byte identity with that unavailable historical URL. See historical_revision_check.json and '
              'windmill_legacy_url_check.json.', '',
              '- `data/think_audit/sources.json`: source URLs and SHA256 values.',
              '- `data/think_audit/stock_sources.json`: provenance for every NYSE/NASDAQ series.',
              '- `runs/think_audit/*.json`: calculations, six witness distances, pair sums, bounds and assumptions.',
              '- `runs/think_audit/*_reconstructed_hyperedges.json`: inspectable US-stock group memberships.',
              '- `hyperbolicity_audit.py`: independent geometry implementation.',
              '- `run_think_audit.py`: reconstruction and preprocessing choices.',
              '- `HOURLY_FORECASTING.md`: the separate hourly-data and live-inference explanation.', '']
    text = '\n'.join(lines)
    Path('THINK_AUDIT.md').write_text(text,encoding='utf-8')
    source = json.loads(Path('data/think_audit/chickenpox.json').read_text())
    graph = nx.Graph()
    graph.add_nodes_from(range(20))
    graph.add_edges_from((a,b) for a,b in source['edges'] if a!=b)
    labels = {index:label for label,index in source['node_ids'].items()}
    w = graph_witness(cp)['witness']
    selected = w['global_vertices']
    fig, axes = plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    positions = nx.spring_layout(graph,seed=7,iterations=150)
    nx.draw_networkx_edges(graph,positions,ax=axes[0],edge_color='#bdc8cf',width=1.3)
    nx.draw_networkx_nodes(graph,positions,ax=axes[0],node_size=100,
                          node_color=['#bd5136' if v in selected else '#43879a' for v in graph])
    nx.draw_networkx_labels(graph,positions,labels=labels,ax=axes[0],font_size=7,
                           bbox={'facecolor':'white','edgecolor':'none','alpha':.75,'pad':1})
    axes[0].set_title('Original county graph: selected witness in orange')
    axes[0].axis('off')
    axes[1].axis('off')
    rows = [[f'd({a.upper()},{b.upper()})',f'{v:g}'] for (a,b),v in w['distances'].items()]
    table = axes[1].table(cellText=rows,colLabels=['Shortest-path distance','Steps'],bbox=[.12,.37,.76,.48])
    table.auto_set_font_size(False); table.set_fontsize(11)
    name_parts = [f'{letter} = {name}' for letter,name in zip('ABCD',w['labels'])]
    names = '     '.join(name_parts[:2]) + '\n' + '     '.join(name_parts[2:])
    axes[1].text(.1,.98,names,va='top',fontsize=10,transform=axes[1].transAxes)
    axes[1].text(.1,.26,'Pair sums: 5, 8, 5\nDelta contribution = (8 - 5) / 2 = 1.5\nMaximum over all 4,845 quadruples = 1.5',
                 fontsize=12,va='top',transform=axes[1].transAxes)
    fig.suptitle('A reproducible hyperbolicity calculation — no forecasting model involved',fontsize=14)
    fig.savefig(ROOT/'chickenpox_calculation.png',dpi=160)
    plt.close(fig)
    print('Wrote THINK_AUDIT.md',flush=True)


if __name__ == '__main__':
    main()
