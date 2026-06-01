"""Tests for the orchestrator idempotency guards (settlement-lag filter + lock)."""
import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("xs_orchestrator", os.path.join(_HERE, "orchestrator.py"))
orch = importlib.util.module_from_spec(_spec)
# orchestrator.py does `from select_engine import compute_target` at import; the
# research dir is already on sys.path via _HERE, so make it importable.
sys.path.insert(0, _HERE)
_spec.loader.exec_module(orch)


def _close(coin, side="long", why="dropped"):
    return (coin, side, why)


def _open(coin, side="long", notion=40.0, price=1.0, why="new"):
    return (coin, side, notion, price, why)


class TestFilterSettling:
    def test_no_attempts_passes_everything(self):
        closes = [_close("ETH")]
        opens = [_open("FET")]
        fc, fo, sup = orch.filter_settling(closes, opens, {}, now=1000.0, grace=600)
        assert fc == closes and fo == opens and sup == []

    def test_recent_coin_is_suppressed(self):
        opens = [_open("FET"), _open("ETH")]
        attempts = {"FET": 950.0}                    # acted 50s ago, grace 600
        fc, fo, sup = orch.filter_settling([], opens, attempts, now=1000.0, grace=600)
        assert [o[0] for o in fo] == ["ETH"]         # FET held back to settle
        assert sup == ["FET"]

    def test_expired_attempt_allows_retry(self):
        opens = [_open("FET")]
        attempts = {"FET": 100.0}                     # acted 900s ago, past 600 grace
        fc, fo, sup = orch.filter_settling([], opens, attempts, now=1000.0, grace=600)
        assert [o[0] for o in fo] == ["FET"] and sup == []

    def test_resize_is_suppressed_on_both_legs(self):
        # a resize emits a close AND an open for the same coin; suppressing one
        # leg but not the other would half-apply the resize.
        closes = [_close("ARB", "short", "resize 40%")]
        opens = [_open("ARB", "short", why="resize")]
        attempts = {"ARB": 990.0}
        fc, fo, sup = orch.filter_settling(closes, opens, attempts, now=1000.0, grace=600)
        assert fc == [] and fo == [] and sup == ["ARB"]

    def test_suppressed_only_lists_coins_we_would_have_touched(self):
        # an attempt for a coin not in this run's actions shouldn't appear
        opens = [_open("FET")]
        attempts = {"FET": 990.0, "DOGE": 990.0}      # DOGE not acted on this run
        fc, fo, sup = orch.filter_settling([], opens, attempts, now=1000.0, grace=600)
        assert sup == ["FET"]


class TestMonitorCloses:
    def _cur(self, **sides):
        return {c: {"side": s, "notional": 40.0} for c, s in sides.items()}

    def test_regime_flat_closes_everything(self):
        cur = self._cur(FET="long", APE="short")
        out = orch.monitor_closes(cur, {"FET": 0.9, "APE": 0.1}, gated=True)
        assert {c for c, _, _ in out} == {"FET", "APE"}
        assert all(why == "regime-flat" for _, _, why in out)

    def test_long_closed_when_rank_decays_below_threshold(self):
        cur = self._cur(FET="long")
        out = orch.monitor_closes(cur, {"FET": 0.30}, gated=False, exit_pct=0.45)
        assert [c for c, _, _ in out] == ["FET"]

    def test_long_held_when_rank_still_strong(self):
        cur = self._cur(FET="long")
        out = orch.monitor_closes(cur, {"FET": 0.80}, gated=False, exit_pct=0.45)
        assert out == []

    def test_short_closed_when_rank_rises_above_upper(self):
        cur = self._cur(APE="short")
        out = orch.monitor_closes(cur, {"APE": 0.70}, gated=False, exit_pct=0.45)
        assert [c for c, _, _ in out] == ["APE"]   # 0.70 > 1-0.45=0.55

    def test_short_held_when_still_weak(self):
        cur = self._cur(APE="short")
        out = orch.monitor_closes(cur, {"APE": 0.20}, gated=False, exit_pct=0.45)
        assert out == []

    def test_inside_dead_band_holds_both_sides(self):
        cur = self._cur(FET="long", APE="short")
        # 0.50 is inside [0.45, 0.55] for both -> hold
        out = orch.monitor_closes(cur, {"FET": 0.50, "APE": 0.50}, gated=False, exit_pct=0.45)
        assert out == []

    def test_unrankable_coin_is_held_not_closed(self):
        cur = self._cur(FET="long")
        out = orch.monitor_closes(cur, {}, gated=False, exit_pct=0.45)   # FET absent from ranks
        assert out == []


class TestLock:
    def test_lock_is_exclusive(self, tmp_path):
        p = str(tmp_path / "x.lock")
        first = orch.acquire_lock(p)
        assert first is not None
        second = orch.acquire_lock(p)        # same process, second flock must fail
        assert second is None
        first.close()                        # releasing lets a later run acquire
        third = orch.acquire_lock(p)
        assert third is not None
        third.close()
