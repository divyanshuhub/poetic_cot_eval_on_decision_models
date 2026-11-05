#!/usr/bin/env python3
"""Run from the repository root: python enable_cot_concurrency.py"""
import ast
from pathlib import Path
import shutil

# TARGET = Path('arc_decision/cot_evaluation.py')
TARGET = Path('src/arc_decision/cot_evaluation.py')
WRAPPER = '''
def evaluate_cot_decisions(
    source, *, config, store, model_identity, client, concurrency=1,
):
    """Bounded parallel HTTP evaluation; only the main thread writes attempts."""
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    from dataclasses import replace
    from queue import Queue, Empty
    import random

    if concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")
    if config.decision_model.lower() == "laya" and concurrency != 1:
        raise ValueError("Laya local inference requires --concurrency 1.")
    if concurrency == 1:
        return _evaluate_cot_decisions_sequential(
            source, config=config, store=store,
            model_identity=model_identity, client=client,
        )

    existing = store.completed_keys("decision")
    if existing and not config.resume and not config.force_regenerate:
        raise FileExistsError("Decision attempts already exist. Use --resume or --force-regenerate.")
    events = Queue()
    stopping = __import__('threading').Event()

    class LedgerProxy:
        def completed_keys(self, stage):
            return set(existing)

        def append_attempt(self, stage, attempt):
            events.put((stage, attempt))

    class WorkerClient:
        def decide(self, state, record):
            if stopping.is_set():
                raise RuntimeError("Evaluation interrupted before request dispatch.")
            return client.decide(state, record)

    def drain():
        while True:
            try:
                stage, attempt = events.get_nowait()
            except Empty:
                break
            store.append_attempt(stage, attempt)

    def run_row(record):
        if stopping.is_set():
            return
        # Each row keeps the original condition ordering and cache semantics.
        row_source = replace(source, records=[record])
        _evaluate_cot_decisions_sequential(
            row_source, config=config, store=LedgerProxy(),
            model_identity=model_identity, client=WorkerClient(),
            _quiet=True,
        )

    records = iter(source.records)
    pool = ThreadPoolExecutor(max_workers=concurrency)
    pending = set()
    progress = tqdm(total=len(source.records), desc="decision evaluation", unit="row")
    try:
        for _ in range(concurrency):
            record = next(records, None)
            if record is None:
                break
            pending.add(pool.submit(run_row, record))
        while pending:
            done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            drain()
            for future in done:
                future.result()
                progress.update(1)
                record = next(records, None)
                if record is not None:
                    pending.add(pool.submit(run_row, record))
    finally:
        stopping.set()
        for future in pending:
            future.cancel()
        # Allow already dispatched requests to finish, then persist their events.
        pool.shutdown(wait=True, cancel_futures=True)
        drain()
        progress.close()

'''

def main():
    original = TARGET.read_text()
    if '_evaluate_cot_decisions_sequential' in original:
        raise SystemExit('Concurrency patch is already installed; no files changed.')
    tree = ast.parse(original)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    node = functions['evaluate_cot_decisions']
    lines = original.splitlines(keepends=True)
    block = ''.join(lines[node.lineno - 1:node.end_lineno])
    block = block.replace('def evaluate_cot_decisions(', 'def _evaluate_cot_decisions_sequential(', 1)
    needle = '    client: DecisionClient,\n) -> None:'
    if needle not in block:
        raise SystemExit('Evaluator signature differs from attached version; no files changed.')
    block = block.replace(needle, '    client: DecisionClient,\n    _quiet: bool = False,\n) -> None:', 1)
    needle = 'tqdm(source.records, desc="decision evaluation", unit="row")'
    if needle not in block:
        raise SystemExit('Evaluation loop differs from attached version; no files changed.')
    block = block.replace(needle, 'tqdm(source.records, desc="decision evaluation", unit="row", disable=_quiet)', 1)
    # Preserve existing retries, adding jitter so simultaneous failures do not retry together.
    block = block.replace('time.sleep(min(2 ** (attempt_number - 1), 30))',
                          'time.sleep(min(2 ** (attempt_number - 1), 30) + __import__("random").uniform(0, 1))')
    updated = ''.join(lines[:node.lineno - 1]) + block + '\n\n' + WRAPPER + ''.join(lines[node.end_lineno:])

    def once(old, new):
        nonlocal updated
        if updated.count(old) != 1:
            raise SystemExit('Expected a unique patch anchor: ' + repr(old) + '; no files changed.')
        updated = updated.replace(old, new, 1)

    once('    request_timeout: float = 60.0,\n    resume: bool = False,',
         '    request_timeout: float = 60.0,\n    concurrency: int = 1,\n    resume: bool = False,')
    once('    source = load_cot_csv(input_csv, dataset_type=dataset_type, split=split, limit=limit)',
         '    if concurrency < 1:\n        raise ValueError("--concurrency must be at least 1.")\n    if decision_model.lower() == "laya" and concurrency != 1:\n        raise ValueError("Laya local inference requires --concurrency 1.")\n    source = load_cot_csv(input_csv, dataset_type=dataset_type, split=split, limit=limit)')
    once('        client=decision_client,\n    )',
         '        client=decision_client,\n        concurrency=concurrency,\n    )')
    once('    parser.add_argument("--max-retries", type=int, default=2)',
         '    parser.add_argument("--concurrency", type=int, default=1, help="Maximum simultaneous HTTP requests; Laya requires 1.")\n    parser.add_argument("--max-retries", type=int, default=2)')
    once('            request_timeout=args.request_timeout,',
         '            request_timeout=args.request_timeout,\n            concurrency=args.concurrency,')
    ast.parse(updated)
    # replace(source, ...) requires the original CSV input dataclass.
    csv_class = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CotCsvInput'), None)
    if csv_class is None or not any('dataclass' in ast.unparse(d) for d in csv_class.decorator_list):
        raise SystemExit('CotCsvInput is not the expected dataclass; no files changed.')
    backup = TARGET.with_suffix('.py.before_concurrency')
    if backup.exists():
        raise SystemExit(f'Backup already exists: {backup}; no files changed.')
    shutil.copy2(TARGET, backup)
    temp = TARGET.with_suffix('.py.tmp')
    temp.write_text(updated)
    temp.replace(TARGET)
    print(f'Patched {TARGET}; original saved to {backup}')
    print('Use --concurrency 10 initially, then increase if API limits allow.')

if __name__ == '__main__':
    main()
