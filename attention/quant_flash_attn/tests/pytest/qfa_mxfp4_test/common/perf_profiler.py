"""msprof collection and strict invocation-to-case/backend association.

The dedicated eager run contains exactly warmup + rounds*repeats QFA calls
per backend. Metadata and conversion operators are excluded by OP Type.
Unexpected/missing task counts fail rather than silently matching by shape.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def parse_tasks(directory):
    paths = sorted(Path(directory).rglob('op_summary_*.csv'))
    if not paths:
        raise ValueError(f'No op_summary CSV under {directory}')
    tasks = []
    for path in paths:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row.get('OP Type', '').strip() != 'QuantFlashAttn':
                    continue
                duration = float(row['Task Duration(us)'])
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError(f'Invalid kernel duration in {path}: {duration}')
                tasks.append(dict(duration_us=duration, start_time_us=float(row['Task Start Time(us)']),
                    task_id=row.get('Task ID'), stream_id=row.get('Stream ID'), op_name=row.get('Op Name'),
                    aicore_time_us=row.get('aicore_time(us)'), cube_utilization=row.get('cube_utilization(%)'),
                    source_csv=str(path)))
    if not tasks:
        raise ValueError('msprof exported zero QuantFlashAttn tasks')
    tasks.sort(key=lambda row: row['start_time_us'])
    return tasks, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def attribute_tasks(records, tasks):
    records = sorted(records, key=lambda record: record['started_unix'])
    expected = sum(record['options']['warmup']+record['options']['rounds']*record['options']['repeats']
                   for record in records for _ in record['options']['backends'])
    if len(tasks) != expected:
        raise ValueError(f'QFA task count {len(tasks)} != expected {expected}; cannot associate safely')
    cursor = 0
    for record in records:
        if record['status'] != 'passed' or record['metric'] not in ('msprof_pending', 'msprof_task_us'):
            raise ValueError('Expected successful dedicated msprof records')
        options = record['options']
        warmup, rounds, repeats = (options[k] for k in ('warmup', 'rounds', 'repeats'))
        for backend in options['backends']:
            result = record['backends'][backend]
            count = warmup+rounds*repeats
            if result.get('profile_invocations') != count:
                raise ValueError('Recorded invocation count differs from timing options')
            selected = tasks[cursor:cursor+count]
            cursor += count
            measured = selected[warmup:]
            values = [r['duration_us'] for r in measured]
            # Match the existing contract: mean per round, then median of rounds.
            samples = [statistics.mean(values[i*repeats:(i+1)*repeats]) for i in range(rounds)]
            median = statistics.median(samples)
            result.update(median_us=median, samples_us=samples, min_us=min(samples), max_us=max(samples),
                spread_pct=(max(samples)-min(samples))/median*100, kernel_samples_us=values,
                warmup_kernel_samples_us=[r['duration_us'] for r in selected[:warmup]],
                profile_task_rows=selected, profile_task_count=count)
        record['metric'] = 'msprof_task_us'
    return records


def save_profile_report(prof_dir, output):
    from .perf_runner import write_record
    from .perf_report import summarize
    output = Path(output)
    paths = sorted(output.glob('*.json'))
    records = [json.loads(p.read_text()) for p in paths]
    if not records:
        raise ValueError(f'No case records under {output}')
    tasks, sources = parse_tasks(prof_dir)
    records = attribute_tasks(records, tasks)
    for record in records:
        record['profiling'] = dict(directory=str(Path(prof_dir).resolve()), csv_sha256=sources,
            association='ordered eager invocations; exactly one QuantFlashAttn task per invocation')
    report = summarize(records)
    if report['complete'] != len(records):
        raise ValueError('Incomplete msprof report')
    # Validate all records before changing any pending records.
    by_name = {r['case']['name']: r for r in records}
    for path in paths:
        original = json.loads(path.read_text())
        write_record(path, by_name[original['case']['name']])
    report_dir = output/'msprof_report'
    report_dir.mkdir(exist_ok=True)
    write_record(report_dir/'summary.json', report)
    with (report_dir/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['rows'][0]))
        writer.writeheader()
        writer.writerows(report['rows'])
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2))
    return report


def compare_baseline(report, path, threshold):
    baseline = json.loads(Path(path).read_text())
    indexed = {(r['name'],r['quant_schema']):r for r in baseline['rows']}
    changes = []
    for row in report['rows']:
        previous = indexed.get((row['name'],row['quant_schema']))
        if previous is None or previous['metric'] != row['metric'] or \
                previous.get('case_fingerprint') != row['case_fingerprint']:
            raise ValueError(f"Missing/incompatible baseline for {row['name']}")
        for backend in ('c4','c8','bf16'):
            if row[backend+'_us'] is None:
                continue
            old = previous.get(backend+'_us')
            if not isinstance(old, (int, float)) or old <= 0:
                raise ValueError(f'Missing baseline backend {backend}')
            changes.append(dict(case=row['name'], backend=backend,
                regression_pct=(row[backend+'_us']/old-1)*100))
    return changes, sum(c['regression_pct'] > threshold for c in changes)


def run_or_parse(config):
    output = Path(config.getoption('--perf-output')).resolve()
    prof_dir = config.getoption('--parse-prof')
    if config.getoption('--msprof'):
        if prof_dir:
            raise ValueError('Select --msprof or --parse-prof, not both')
        if not config.getoption('--run-perf'):
            raise ValueError('--msprof requires --run-perf')
        if list(output.glob('*.json')) or (output/'profile').exists():
            raise ValueError('Use a fresh perf-output directory for msprof collection')
        output.mkdir(parents=True, exist_ok=True)
        prof_dir = output/'profile'
        args = [arg for arg in sys.argv[1:] if arg != '--msprof']
        args += ['--perf-mode=msprof']
        command = ['msprof', '--output='+str(prof_dir), '--aic-metrics=PipeUtilization',
                   sys.executable, '-m', 'pytest', *args]
        (output/'msprof_command.txt').write_text(json.dumps(command, indent=2)+'\n')
        env = dict(os.environ, QFA_MSPROF_CHILD='1')
        print('Collecting msprof into', prof_dir, flush=True)
        code = subprocess.call(command, env=env)
        if code:
            raise RuntimeError(f'msprof/pytest failed with exit code {code}; raw output retained at {output}')
    report = save_profile_report(prof_dir, output)
    baseline = config.getoption('--perf-baseline')
    if baseline:
        changes, failed = compare_baseline(report, baseline, config.getoption('--perf-threshold'))
        (output/'msprof_report'/'baseline.json').write_text(json.dumps(changes, indent=2)+'\n')
        if failed:
            print(f'{failed} backend measurements exceed the regression threshold')
            return 1
    return 0
