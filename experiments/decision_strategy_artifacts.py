"""Explicit, bounded artifact indexing and locally verified multipart bundles."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from experiments.strategy_search.storage import atomic,read,sha,utc
from experiments.decision_strategy.config import STUDY,SCOPES


def index(output):
    output=Path(output);repo=Path(__file__).parents[1];protocol=read(output/'protocol.json')
    definitions=read(output/'model_package_references.json');records=[]
    for scope in SCOPES:
        root=STUDY/'cache'/scope;meta=read(root/'market.json')
        from experiments.decision_strategy.data import guard_dates
        guard_dates(meta['dates'],scope,protocol)
        records.append(dict(scope=scope,metadata=str(root/'market.json'),metadata_sha256=sha(root/'market.json'),
            declared_safe_files=meta['files'],authorized_dates=meta['dates'],
            raw_bar_slice=[meta['begin'],meta['end']],raw_archive=str(Path(protocol['binding']['market_root'])/'1m/bars.npy'),
            full_archive_hashed=False,reason='Shared archive contains protected sessions: only guarded cohort slices may be touched.'))
    paths=dict(study=str(output),repository=str(repo),strategies=str(repo/'experiments/decision_strategy'),
        frozen_strategy_source=str(output/('source_'+protocol['binding']['version'].replace('decision_',''))),reproduction_source=str(output/'reproduction_source'),
        input_metadata=records,forecast_definition=definitions['definition'],forecast_factory=definitions['factory'],
        forecast_runtime=definitions['runtime'],frozen_fit_root=definitions['frozen_fits_root'],
        complete_frozen_forecasts=[dict(scope=s,manifest=str(STUDY/'cache'/s/'forecasts.json'),
            HYB_LN=str(STUDY/'cache'/s/'HYB_LN_ensemble.npy'),LN4=str(STUDY/'cache'/s/'LN4_ensemble.npy')) for s in SCOPES],
        checkpoint_and_teacher_references=str(output/'model_package_references.json'),
        reusable_quote_cache=str(output/'quotes'),etf_bars=str(STUDY/'closing_momentum_followup_2026-10-26/etf_cache/bars'),
        raw_historical_classification_evidence=str(STUDY/'decision_strategy_execution_2026-10-27/metadata_sources'),
        charts=str(output/'cost_curves.png'),html_report=str(output/'report.html'),ledger=str(output/'trial_ledger.csv'),
        paired_uncertainty=str(output/'paired_controls.json'),attempt_history=str(output/'attempt_history.json'),
        per_trial_tapes=str(output/'trials/<scope>__<recipe>__<gate>__<scenario>/minute_tapes.npz'),
        per_trial_csv='minute_nav.csv.gz,minute_holdings_sparse.csv.gz,daily_evaluation.csv.gz,decisions.csv.gz,orders.csv.gz,fills.csv.gz; shadow_ files preserve unscaled books',
        sparse_holdings='Every omitted minute/ticker holding equals exactly zero; dense integer matrix remains in NPZ.',
        neural_value_access='No forecast values opened unless preregistered D06 base gate becomes eligible.')
    atomic(output/'FILE_PATHS.json',paths)
    (output/'FILE_PATHS.txt').write_text(json.dumps(paths,indent=2),encoding='utf-8')
    source=output/'reproduction_source'
    files={str(Path('experiments/decision_strategy')/name):repo/'experiments/decision_strategy'/name for name in protocol['binding']['source']}
    files.update({name:repo/name for name in protocol['binding']['dependencies']})
    files.update({str(p.relative_to(repo)):p for p in (repo/'experiments').glob('decision_strategy_*.py')})
    files.update({str(p.relative_to(repo)):p for p in (repo/'test').glob('test_decision_strategy*.py')})
    files['uv.lock']=repo/'uv.lock';files['experiments/decision_strategy/README.txt']=repo/'experiments/decision_strategy/README.txt'
    for name,p in files.items():
        dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha(dest)!=sha(p):raise ValueError('Changed reproduction snapshot:'+name)
        if not dest.exists():shutil.copyfile(p,dest)
    atomic(source/'manifest.json',dict(files={name:sha(source/name) for name in sorted(files)},
        no_raw_data_or_credentials=True,protocol_sha256=sha(output/'protocol.json'),line_endings_preserved=True))
    evidence_root=STUDY/'decision_strategy_execution_2026-10-27/metadata_sources'
    evidence_names=('acquisition.json','schwab_us_2024.html','sap_business_2024.html','sanofi_2023.html','gsk_2023.pdf')
    evidence=[]
    for name in evidence_names:
        p=evidence_root/name
        evidence.append(dict(name=name,sha256=sha(p),size=p.stat().st_size,original=str(p),local_reference_only=True))
    atomic(output/'classification_source_evidence/manifest.json',dict(files=evidence,
        packaged_evidence='Matched classification facts and primary-source links are in causal_classifications.json; complete third-party filings remain hash-bound local references.',
        other_primary_disclosure_links_and_read_limitations='causal_classifications.json rows;9 manual business judgments are explicitly distinguished',
        rejected_sources_preserved_at=str(evidence_root),later_classification_values_used=False))
    atomic(output/'input_data_references.json',dict(created_UTC=utc(),cohorts=records,
        licensed_market_archives_are_local_references=True,protected_outcomes_accessed=False,
        checkpoint_bytes_duplicated=False,raw_neural_forecasts_duplicated=False))
    return paths


def forward_spec(output):
    output=Path(output);protocol=read(output/'protocol.json')
    manifest=protocol['binding']['protected'];dates=sorted(manifest['reserved_session_dates'])
    selection=read(output/'summary.json')['qualified']
    result=dict(created_UTC=utc(),status='inactive_unscheduled_specification_only',collect_or_trade_authorized=False,
        source_frozen_UTC=protocol['frozen_UTC'],source_sha256=sha(output/'protocol.json'),
        protected_manifest_sha256=protocol['binding']['protected_sha256'],reserved_through=dates[-1],
        earliest_permitted_new_session=manifest['next_eligible_session'],requires_after_source_freeze_and_actual_reservation_end=True,
        historical_candidates=selection,policy='Only a historically qualified frozen policy can be nominated; an empty set means no strategy nomination.',
        prospective_requirement='60 consecutive newly observed eligible sessions, with20 initial cash volatility warmup sessions reported explicitly; no peeking, no resetting because outcomes disappoint.',
        locked_risk_contract=protocol['binding']['sizing'],locked_execution_contract=protocol['binding']['execution'],
        qualification_contract=protocol['binding']['qualification'],
        monitoring='Record intended/filled orders,as-of SIP coverage,fees,minute holdings/NAV,availability and every constraint failure; mandatory liquidation continues immediately.',
        future_changes='Any strategy or threshold change needs a new source freeze before new outcomes. Never open reserved outcomes as a replacement holdout.',
        collection_started=False,automation_created=False,orders_submitted=False)
    atomic(output/'forward_test_specification.json',result);return result


def package(output,limit=64_000_000):
    """Independent ZIP parts, exact SHA256 member checks and CRC readback.

    References to existing licensed price/forecast archives avoid copying large
    shared histories containing protected dates. Trial outputs and quote cache
    acquired by this study are included. No credentials/checkpoints enter ZIPs.
    """
    output=Path(output);dest=output/'bundles';dest.mkdir(exist_ok=True)
    protocol_file=output/'protocol.json'
    version=read(protocol_file)['binding']['version'].replace('decision_','') if protocol_file.exists() else 'v7'
    forbidden={'bundles','cloud_uploads','__pycache__'}
    excluded={'artifact_manifest.json','cloud_upload_verification.json','FINAL_VERIFICATION.json','progress.json','RESUME_NOTES.txt'}
    files=[p for p in sorted(output.rglob('*')) if p.is_file() and not forbidden.intersection(p.relative_to(output).parts)
        and p.name not in excluded and p.suffix not in ('.pyc','.tmp','.partial')]
    if any(p.name=='.env' or p.suffix in ('.pt','.pth') for p in files):raise ValueError('Disallowed secret/checkpoint filename')
    records=[];parts=[];batch=[];size=0
    def save(items,number):
        name='decision_strategy_'+version+'_part'+str(number).zfill(3)+'.zip';p=dest/name
        estimate=sum(q.stat().st_size for q in items)
        members={q.relative_to(output).as_posix():dict(size=q.stat().st_size,sha256=sha(q)) for q in items}
        if not p.exists():
            if shutil.disk_usage(dest).free-estimate<10_000_000_000:raise RuntimeError('10GB reserve')
            with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
                for q in items:z.write(q,q.relative_to(output).as_posix())
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:raise ValueError('Archive CRC failure')
            if set(z.namelist())!=set(members):raise ValueError('Archive member set differs')
            for member,expected in members.items():
                digest=hashlib.sha256();length=0
                with z.open(member) as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk);length+=len(chunk)
                if digest.hexdigest()!=expected['sha256'] or length!=expected['size']:raise ValueError('Archive content readback differs')
        parts.append(dict(name=name,path=str(p),size=p.stat().st_size,sha256=sha(p),members=len(members),CRC_verified=True,member_SHA256_verified=True))
        records.extend(dict(path=member,part=name,**v) for member,v in members.items())
        print('VERIFIED_PART',name,p.stat().st_size,len(members),flush=True)
    for p in files:
        if batch and size+p.stat().st_size>limit:save(batch,len(parts)+1);batch=[];size=0
        batch.append(p);size+=p.stat().st_size
    if batch:save(batch,len(parts)+1)
    result=dict(created_UTC=utc(),source_root=str(output),parts=parts,files=records,
        verification='Every archive SHA256,CRC and decompressed member SHA256/size checked locally.',
        excluded_mutable_delivery_receipts=sorted(excluded),large_existing_input_archives='Exact guarded local references in input_data_references.json; shared protected history never copied.',
        cloud_checksum_status='Pending; Drive metadata readback is distinct from cloud content checksum.')
    atomic(output/'artifact_manifest.json',result);return result


def verify_completion(output,require_cloud=True):
    """Fail if a required deliverable is absent, pending, changed or incomplete."""
    import xml.etree.ElementTree as ET
    from experiments.decision_strategy_parallel import plan
    from experiments.decision_strategy_delivery import validate_files
    output=Path(output);errors=[];integrity=validate_files(output)
    proxies=read(output/'proxy_results.json')['rows'];quotes=read(output/'quote_results.json')
    expected={'__'.join(row) for row in plan()};actual={r['key'] for r in proxies}
    if actual!=expected or len(proxies)!=315:errors.append('Incomplete/duplicate frozen proxy plan')
    selection=read(output/'selection.json');dispositions=read(output/'design_dispositions.json')
    fitted=read(output/'fitted_data_audit.json')
    if not fitted['passed'] or len(fitted['proofs'])!=12:errors.append('Saved fit/OOF ownership audit incomplete')
    if dispositions['pending_required_work']:errors.append('Eligible design work remains pending')
    if {r['design'] for r in dispositions['designs']}!={'D'+str(k).zfill(2) for k in range(1,11)}:errors.append('Missing design disposition')
    if not selection['diagnostics'] and (quotes['rows'] or not quotes['empty_set_is_not_coverage']):errors.append('Misrepresented empty quote set')
    execution=read(output/'independent_execution_verification.json');attribution=read(output/'attribution_verification.json')
    n=len(proxies)+len(quotes['rows'])
    if execution['partial'] or execution['trial_count']!=n or not execution['independently_reconciled']:errors.append('Independent execution audit incomplete')
    if len(attribution['proofs'])!=n or not attribution['passed']:errors.append('Attribution audit incomplete')
    for proof in execution['proofs']:
        if sha(output/'trials'/proof['key']/'independent_execution.json')!=proof['sha256']:errors.append('Changed independent audit:'+proof['key'])
    for proof in attribution['proofs']:
        if sha(output/'trials'/proof['key']/'independent_attribution.json')!=proof['sha256']:errors.append('Changed attribution:'+proof['key'])
        for name,evidence in proof['csv_exports'].items():
            p=output/'trials'/proof['key']/name
            if sha(p)!=evidence['sha256'] or p.stat().st_size!=evidence['size']:errors.append('Changed/incomplete CSV export:'+proof['key']+'/'+name)
    suite=ET.parse(output/'tests.xml').getroot().find('testsuite')
    if suite is None or int(suite.get('errors',1)) or int(suite.get('failures',1)):errors.append('Repository tests failed or unverified')
    commit=read(output/'source_delivery_verification.json')
    if not commit['remote_commit_verified'] or not commit['only_reusable_source_tests_docs']:errors.append('Source push unverified')
    artifacts=read(output/'artifact_manifest.json')
    for part in artifacts['parts']:
        p=Path(part['path'])
        if p.stat().st_size!=part['size'] or sha(p)!=part['sha256']:errors.append('Changed archive:'+part['name'])
    forward=read(output/'forward_test_specification.json')
    if forward['collection_started'] or forward['automation_created'] or forward['orders_submitted']:errors.append('Forward specification improperly activated')
    cloud=read(output/'cloud_upload_verification.json') if require_cloud else dict(passed=None)
    if require_cloud and not cloud['passed']:errors.append('Cloud uploads unverified')
    required={'report.html','summary.json','trial_ledger.csv','cost_curves.png','artifact_manifest.json','FILE_PATHS.txt','judge_questions.html','forward_test_specification.json'}
    if require_cloud:
        names={r['name'] for r in cloud['files'] if r['metadata_verified']}
        if not required.issubset(names) or not {p['name'] for p in artifacts['parts']}.issubset(names):errors.append('Required cloud files missing')
    summary=read(output/'summary.json')
    result=dict(verified_UTC=utc(),passed=not errors,errors=errors,proxy_trials=len(proxies),quote_trials=len(quotes['rows']),
        qualified=summary['qualified'],current_protocol_sha256=sha(output/'protocol.json'),
        repository_tests=dict(suite.attrib) if suite is not None else None,
        local_integrity=integrity,independent_execution_trials=execution['trial_count'],attributed_trials=len(attribution['proofs']),
        fitted_recipes_verified=len(fitted['proofs']),
        designs=10,pending_design_work=dispositions['pending_required_work'],protected_future_outcomes_accessed=False,
        source_commit=commit['commit'],remote=commit['remote'],cloud_metadata_verified=cloud['passed'],
        cloud_content_checksum=cloud.get('cloud_content_checksum_status','not evaluated'),
        drive_desktop_sync_claimed=False,sharing='Existing anyone-with-link reader setting preserved; no permission mutation.',
        forward_collection=False,live_orders=False,archive_parts=len(artifacts['parts']),archive_member_hashes_verified=True,
        risk_failures_retained=True,empty_quote_receipts_are_not_coverage=True)
    atomic(output/'FINAL_VERIFICATION.json',result)
    if errors:raise ValueError(errors)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['index','forward','bundle','verify']);p.add_argument('--output',default=str(STUDY/'decision_strategy_execution_2026-10-27_v7'));a=p.parse_args()
    {'index':index,'forward':forward_spec,'bundle':package,'verify':verify_completion}[a.command](a.output)
