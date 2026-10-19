"""Fail if release candidates diverge from the reviewed text-file allowlist.

This is a repeatable screening check, not a substitute for manual source review.
"""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = {
    '.gitattributes', '.gitignore', '.github/workflows/synthetic-tests.yml',
    'README.md', 'NOTICE.md', 'pyproject.toml', 'requirements.txt',
    'docs/ARCHITECTURE.md', 'docs/DATA_PROTOCOL.md', 'docs/PROVENANCE.md',
    'docs/RELEASE_AUDIT.md',
    'scripts/audit_release.py', 'scripts/synthetic_demo.py',
    'scripts/calibrate_personal_activity_highlow.py',
    'scripts/predict_personal_activity_highlow.py',
    'src/eeg_engagement/__init__.py', 'src/eeg_engagement/local_data.py',
    'src/eeg_engagement/synthetic.py',
    'src/eeg_engagement/benchmarking/__init__.py',
    'src/eeg_engagement/benchmarking/models.py',
    'src/eeg_engagement/benchmarking/data_scope.py',
    'src/eeg_engagement/calibrated_highlow/__init__.py',
    'src/eeg_engagement/calibrated_highlow/v1.py',
    'src/eeg_engagement/preprocessing/__init__.py',
    'src/eeg_engagement/preprocessing/signal.py',
    'src/eeg_engagement/features/__init__.py',
    'src/eeg_engagement/features/bandpower.py',
    'tests/conftest.py', 'tests/test_data_scope.py',
    'tests/test_personal_activity_highlow.py', 'tests/test_splits_and_features.py',
}
PATTERNS = {
    'credential': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})'),
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'absolute user path': re.compile(r'(?:\b[A-Za-z]:[\\/]|/(?:home|Users)/[^/\s]+/)'),
    'external drive link': re.compile(r'https?://(?:drive|docs)\.google\.com/'),
    'email address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
}


def audit():
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT,
    ).decode().split('\0')
    candidates = set(filter(None, names))
    errors = []
    if candidates != ALLOWLIST:
        errors.append(f'Allowlist mismatch: extra={sorted(candidates - ALLOWLIST)}, missing={sorted(ALLOWLIST - candidates)}')
    for name in sorted(candidates):
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            errors.append(f'Unsafe release path: {name}')
            continue
        try:
            raw = path.read_bytes()
            content = raw.decode('utf-8')
        except (OSError, UnicodeError):
            errors.append(f'Not a readable UTF-8 text file: {name}')
            continue
        if b'\0' in raw or len(raw) > 100_000:
            errors.append(f'Unexpected binary or large content: {name}')
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                errors.append(f'{label}: {name}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(candidates)} reviewed text files; allowlist and privacy screening passed.')


if __name__ == '__main__':
    audit()
