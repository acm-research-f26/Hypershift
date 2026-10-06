"""Audit available THINK source datasets under explicitly declared reconstructions.

This script does not claim possession of THINK's missing processed incidence
matrices or feature tensors. All choices and mathematical bounds are recorded.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from hyperbolicity_audit import (delta_metric, euclidean_distances, graph_audit,
    adjacency_from_edges, group_adjacency, neighborhood_groups, merge_dice)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/think_audit'))
    parser.add_argument('--out', type=Path, default=Path('runs/think_audit'))
    parser.add_argument('--section', choices=['public','stocks'], default='public')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}

    def save(key, result, assumption):
        result['assumption'] = assumption
        results[key] = result
        (args.out / f'{key}.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=lambda v:v.item()), encoding='utf-8')
        print(key, {k:v for k,v in result.items() if k in ['global_delta','max_component_lower_bound','max_component_upper_bound','component_count','delta','relative_lower_bound','exact','diameter']}, flush=True)

    def feature(key, vectors, assumption, labels):
        result = delta_metric(euclidean_distances(vectors))
        if result['witness']:
            result['witness']['labels'] = [str(labels[i]) for i in result['witness']['vertices']]
        save(key, result, assumption)

    if args.section == 'public':
        c = json.loads((args.data / 'chickenpox.json').read_text())
        labels = sorted(c['node_ids'], key=c['node_ids'].get)
        x = np.array(c['FX'], dtype=float)
        a = adjacency_from_edges(x.shape[1], c['edges'])
        save('CPox_source_graph', graph_audit(a, labels), 'Public county adjacency, undirected, self-loops removed; this is not the merged THINK hypergraph.')
        groups = neighborhood_groups(a)
        for s in [1,2]:
            save(f'CPox_neighborhood_s{s}', graph_audit(group_adjacency(len(a), groups, s), labels), f'Closed neighborhoods, no merging, retain all 20 columns, s={s}.')
        for threshold in [.5,.75]:
            merged = merge_dice(groups, threshold)
            result = graph_audit(group_adjacency(len(a), merged), labels)
            result['hyperedges'] = [sorted(g) for g in merged]
            save(f'CPox_dice_{threshold}', result, f'Greedy union of most similar groups with Dice >= {threshold}; s=1. Assumed rule, not published setting.')
        merged = merge_dice(groups,.5,False)
        result = graph_audit(group_adjacency(len(a), merged), labels)
        result['hyperedges'] = [sorted(g) for g in merged]
        save('CPox_literal_low_dice_0.5', result, 'Literal below-threshold merge interpretation: greedily union smallest Dice <0.5; s=1. Sensitivity check only.')
        feature('CPox_raw_trajectories', x.T, f'20 county points, each represented by all {len(x)} published FX values. These are already column-standardized, not raw case counts.', labels)
        z = (x-x.mean(0))/np.maximum(x.std(0),1e-12)
        feature('CPox_standardized_trajectories', z.T, 'Same 20 trajectories standardized separately per county across the full dataset; descriptive geometry, not a training pipeline.', labels)

        w = json.loads((args.data / 'windmill_output.json').read_text())
        x, weights = np.array(w['block'],dtype=float), np.array(w['weights'])
        edges = np.array(w['edges'], dtype=int)
        labels = list(range(x.shape[1]))
        # All supplied weights are positive: literal adjacency is complete.
        for threshold in [0,.5]:
            a = adjacency_from_edges(len(labels), edges[weights > threshold])
            save(f'WMill_weight_gt_{threshold}_neighborhood_s1', graph_audit(group_adjacency(len(a), neighborhood_groups(a)), labels),
                 f'Undirected edges with weight > {threshold}, then closed neighborhoods without merging, s=1. Threshold .5 is a sensitivity assumption; 0 keeps every positive supplied edge.')
        feature('WMill_raw_trajectories', x.T, f'319 windmill points, each represented by all {len(x)} raw hourly observations.', labels)
        z = (x-x.mean(0))/np.maximum(x.std(0),1e-12)
        feature('WMill_standardized_trajectories', z.T, 'Same 319 trajectories standardized separately per turbine over the full dataset; descriptive geometry.', labels)
        del w, x, z

        for event in ['rg17','uo17']:
            source = json.loads((args.data / f'twitter_tennis_{event}.json').read_text())
            summaries = []
            for t in range(source['time_periods']):
                snapshot = source[str(t)]
                n = len(snapshot['y'])
                a = adjacency_from_edges(n, snapshot['edges'])
                result = graph_audit(group_adjacency(n, neighborhood_groups(a)))
                # Keep witnesses of nontrivial components, summarize isolated vertices.
                result['components'] = [r for r in result['components'] if r['n'] >= 4]
                result['snapshot'] = t
                summaries.append(result)
                if t % 30 == 0:
                    print('DTT',event,'snapshot',t,flush=True)
            result = {'snapshot_count': len(summaries), 'disconnected_snapshots': sum(r['component_count']>1 for r in summaries),
                      'max_component_lower_bound': max(r['max_component_lower_bound'] for r in summaries),
                      'max_component_upper_bound': max(r['max_component_upper_bound'] for r in summaries),
                      'all_components_exact': all(r['all_components_exact'] for r in summaries), 'snapshots': summaries}
            save(f'DTT_{event}_neighborhood_s1', result, 'Every supplied timestamp, symmetrized mentions, unmerged closed neighborhoods, s=1. Disconnected components evaluated separately. Event and aggregation rule are not specified by THINK.')
    else:
        for market in ['NYSE','NASDAQ']:
            tickers = (args.data / f'{market}_tickers_qualify_dr-0.98_min-5_smooth.csv').read_text().splitlines()
            ids = {t:i for i,t in enumerate(tickers)}
            industry = json.loads((args.data / f'{market}_industry_ticker.json').read_text())
            groups = [set(ids[t] for t in members if t in ids) for name,members in industry.items() if name != 'n/a']
            groups = [g for g in groups if g]
            wiki_ids = {}
            for line in (args.data / f'{market}_wiki.csv').read_text().splitlines():
                ticker, wiki = line.split(',')
                if wiki != 'unknown' and ticker in ids:
                    wiki_ids[wiki] = ids[ticker]
            selected = {line.split()[0] for line in (args.data / 'selected_wiki_connections.csv').read_text().splitlines()}
            connections = json.loads((args.data / f'{market}_connections.json').read_text())
            first_order = defaultdict(set)
            for src, destinations in connections.items():
                if src not in wiki_ids:
                    continue
                for dst, paths in destinations.items():
                    if dst not in wiki_ids:
                        continue
                    for path in paths:
                        if '_'.join(path) not in selected:
                            continue
                        if len(path)==1:
                            first_order[(src,path[0])].update([wiki_ids[src],wiki_ids[dst]])
                        elif len(path)==2:
                            groups.append({wiki_ids[src],wiki_ids[dst]})
            groups.extend(first_order.values())
            groups = [set(g) for g in sorted(set(tuple(sorted(g)) for g in groups))]
            (args.out / f'{market}_reconstructed_hyperedges.json').write_text(json.dumps([[tickers[i] for i in sorted(g)] for g in groups]),encoding='utf-8')
            for s in [1,2]:
                result = graph_audit(group_adjacency(len(tickers),groups,s),tickers)
                result['hyperedge_count'] = len(groups)
                save(f'{market}_relations_s{s}', result, f'Original industry groups excluding n/a; original selected Wikidata paths, grouped source+relation for first order, pairs for second order; deduplicated memberships; s={s}. THINK incidence file unavailable.')
            panel = np.stack([np.loadtxt(args.data / market / f'{market}_{t}_1.csv',delimiter=',') for t in tickers])
            if market=='NASDAQ':
                panel = panel[:,:-1]  # Matches the cited RSR loader, not an inferred THINK setting.
            raw = panel[:,:,1:]
            missing = np.isclose(raw,-1234)
            filled = np.where(missing,1.1,raw)  # Explicitly mirror cited RSR preprocessing.
            result_shape = {'nodes': len(tickers), 'timesteps': raw.shape[1], 'features_per_time': raw.shape[2], 'sentinel_values_replaced': int(missing.sum())}
            (args.out / f'{market}_panel_shape.json').write_text(json.dumps(result_shape,indent=2),encoding='utf-8')
            feature(f'{market}_RSR_feature_trajectories', filled.reshape(len(tickers),-1),
                    'One point per stock: flatten all five processed RSR feature trajectories. NASDAQ last row removed; sentinel -1234 replaced with 1.1 as in RSR loader. THINK temporal feature tensor unspecified.', tickers)
            z = (filled-filled.mean(1,keepdims=True))/np.maximum(filled.std(1,keepdims=True),1e-12)
            feature(f'{market}_standardized_trajectories', z.reshape(len(tickers),-1),
                    'Alternative: standardize each of the five per-stock trajectories before flattening. Full dataset used for descriptive geometry only.', tickers)
    (args.out / f'{args.section}_index.json').write_text(json.dumps(list(results),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
