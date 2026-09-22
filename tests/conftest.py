"""
Autouse fixture redirecting every self_review data path to an isolated
location, for every test in the suite - not just the ones that know they
touch issue logging.

Found the hard way: adding real file-write side effects inside
ollama_client.py's grounding-nudge branch (assistant/self_review/issue_log.py)
made pre-existing, otherwise-pure tests in test_ollama_client.py (written
before that logging existed, exercising the exact same code path with no
reason to expect a disk write) silently append real entries to the actual
data/issues_log.jsonl in the repo - confirmed by finding that file polluted
with test data after a normal test run. A single global fixture, applied to
every test automatically, is the systemic fix - it's not enough to remember
to mock this per-test, since any future test exercising this code path has
the same exposure by default.

Uses a session-scoped base directory (one real mkdir for the whole run) and
pure path arithmetic per test (no I/O) rather than the per-test `tmp_path`
fixture - found that using `tmp_path` here, applied automatically to all
~5000+ tests, forced a real directory creation for every single one
regardless of whether it touches self_review at all, and roughly tripled
the whole suite's runtime (~20s -> ~55s, confirmed by measuring both). The
actual writes this isolates only happen in the small minority of tests that
call issue_log functions directly - issue_log itself creates its parent
directory lazily on first write, so no eager mkdir is needed here either.
"""

import hashlib

import pytest

from assistant.self_review import issue_log
from assistant.tools import web


@pytest.fixture(scope="session")
def _self_review_base_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("self_review_isolation")


@pytest.fixture(autouse=True)
def _isolate_self_review_data(_self_review_base_dir, monkeypatch, request):
    # A hash of the full node id (not request.node.name) keeps this
    # filesystem-safe - raw parametrize IDs can contain characters like '/'
    # that aren't valid in a path segment.
    safe_name = hashlib.md5(request.node.nodeid.encode()).hexdigest()
    test_dir = _self_review_base_dir / safe_name
    monkeypatch.setattr(issue_log, "LOG_PATH", test_dir / "issues_log.jsonl")
    monkeypatch.setattr(issue_log, "REPORTS_DIR", test_dir / "daily_reports")
    monkeypatch.setattr(issue_log, "LAST_CHECKIN_PATH", test_dir / "last_checkin_date.txt")


@pytest.fixture(autouse=True)
def _no_real_cdp_calls(monkeypatch):
    # open_website's new-tab path tries a real CDP call (browser._new_tab)
    # against 127.0.0.1:9222 before falling back to the older
    # browser_state-based approach - the right behavior in production (see
    # web.py), but in a normal test run nothing is actually listening on
    # that port, so every un-mocked call was paying a real ~2s
    # connection-refused delay. Same systemic-fixture reasoning as
    # _isolate_self_review_data above: default every test to the "no
    # CDP-debugged browser reachable" outcome instantly, and let any test
    # that specifically wants to exercise the CDP path override it with its
    # own patch.object(web, "_new_tab", ...), same as it already does today.
    def _no_cdp_browser(url):
        raise ConnectionError("no debugged browser (test default)")

    monkeypatch.setattr(web, "_new_tab", _no_cdp_browser)
