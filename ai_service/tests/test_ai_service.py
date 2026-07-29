"""
test_ai_service.py
------------------
Pytest test suite for the niyojak AI service (Python layer only).

Coverage areas
--------------
  1. train_model.py   — dataset generation shape, feature columns, score ranges,
                        scenario edge-case invariants, determinism
  2. feature_store.py — NodeMetricsWindow aggregation, sliding-window semantics,
                        spike-rate logic, FeatureStore fallback behaviour,
                        feature dict key/order contract
  3. model.py         — heuristic scorer correctness, clamping, feature-vector
                        ordering, model source reporting, XGBoost path (mock)
  4. main.py          — FastAPI /score endpoint request handling, response schema,
                        and score range validation via TestClient

Run with:
    pytest ai_service/tests/test_ai_service.py -v
"""

import sys
import os
import math
import pickle
import tempfile
import threading
import unittest.mock as mock

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Add ai_service/app and ai_service/train to sys.path so bare imports
# resolve both at runtime and for static-analysis tools (Pyright / Pylance).
# ---------------------------------------------------------------------------
_ai_service_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _subdir in ("app", "train"):
    _path = os.path.join(_ai_service_root, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import train_model  # noqa: E402  # type: ignore[import-not-found]
import feature_store as fs_module  # noqa: E402  # type: ignore[import-not-found]
from feature_store import NodeMetricsWindow, FeatureStore, WINDOW_SIZE  # noqa: E402  # type: ignore[import-not-found]
import model as model_module  # noqa: E402  # type: ignore[import-not-found]
from model import NodeScorer, FEATURE_COLUMNS  # noqa: E402  # type: ignore[import-not-found]


# ===========================================================================
# Section 1 — train_model.py
# ===========================================================================

class TestGenerateDataset:
    """Dataset generation shape, columns, score ranges, and edge-case invariants."""

    # -----------------------------------------------------------------------
    # Use a small n_samples so tests stay fast; 1000 still exercises all 10
    # scenario branches evenly (100 samples each).
    # -----------------------------------------------------------------------
    N = 1_000

    @pytest.fixture(scope="class")
    @classmethod
    def df(cls, tmp_path_factory):
        return train_model.generate_dataset(n_samples=cls.N)

    # --- Shape ---

    def test_row_count(self, df):
        """Dataset must have exactly n_samples rows."""
        assert len(df) == self.N

    def test_column_count(self, df):
        """Dataset must have all 11 feature columns plus the score label."""
        assert len(df.columns) == len(train_model.FEATURE_COLUMNS) + 1

    def test_all_feature_columns_present(self, df):
        """Every column in FEATURE_COLUMNS must appear in the dataset."""
        for col in train_model.FEATURE_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_score_column_present(self, df):
        assert "score" in df.columns

    def test_feature_columns_match_model(self):
        """train_model.FEATURE_COLUMNS must be identical to model.FEATURE_COLUMNS."""
        assert train_model.FEATURE_COLUMNS == FEATURE_COLUMNS

    # --- No bad values ---

    def test_no_nans(self, df):
        assert not df.isnull().any().any(), "Dataset contains NaN values"

    def test_no_infs(self, df):
        numeric = df.select_dtypes(include=[float, int])
        assert not np.isinf(numeric.values).any(), "Dataset contains Inf values"

    # --- Score range ---

    def test_score_global_min(self, df):
        assert df["score"].min() >= 0.0

    def test_score_global_max(self, df):
        assert df["score"].max() <= 100.0

    def test_score_spans_full_range(self, df):
        """The full dataset should cover both near-zero and near-100 scores."""
        assert df["score"].min() < 20.0, "Expected some low-score (stressed) samples"
        assert df["score"].max() > 80.0, "Expected some high-score (healthy) samples"

    # --- Feature value sanity ---

    def test_cpu_mean_bounded(self, df):
        assert df["cpu_mean"].between(0.0, 1.0).all()

    def test_mem_mean_bounded(self, df):
        assert df["mem_mean"].between(0.0, 1.0).all()

    def test_cpu_spike_rate_bounded(self, df):
        assert df["cpu_spike_rate"].between(0.0, 1.0).all()

    def test_cpu_max_gte_cpu_mean(self, df):
        """cpu_max must never be less than cpu_mean (within float tolerance)."""
        diff = df["cpu_max"] - df["cpu_mean"]
        assert (diff >= -1e-9).all(), "cpu_max < cpu_mean found in some rows"

    def test_mem_max_gte_mem_mean(self, df):
        diff = df["mem_max"] - df["mem_mean"]
        assert (diff >= -1e-9).all()

    def test_load_max_gte_load_mean(self, df):
        diff = df["load_max"] - df["load_mean"]
        assert (diff >= -1e-9).all()

    def test_net_rx_mean_nonnegative(self, df):
        assert (df["net_rx_mean"] >= 0).all()

    def test_net_tx_mean_nonnegative(self, df):
        assert (df["net_tx_mean"] >= 0).all()

    # --- Scenario-specific score-band coverage ---

    def test_healthy_idle_band_present(self, df):
        """Scenario 1: at least some samples should have score >= 88."""
        assert (df["score"] >= 88).any()

    def test_stressed_band_present(self, df):
        """Scenarios 9 & 10: at least some samples should have score <= 12."""
        assert (df["score"] <= 12).any()

    def test_memory_pressure_edge_case(self, df):
        """
        Memory-pressure scenario: rows with high mem_mean but low cpu_mean.
        With n=1000 (100 per scenario) this should always exist.
        """
        mask = (df["mem_mean"] > 0.75) & (df["cpu_mean"] < 0.40)
        assert mask.any(), "No memory-pressure edge-case rows found"

    def test_cpu_burst_edge_case(self, df):
        """CPU burst scenario: high cpu_std despite moderate cpu_mean."""
        mask = (df["cpu_std"] > 0.10) & (df["cpu_mean"] < 0.65)
        assert mask.any(), "No CPU-burst/flapping rows found"

    def test_network_bound_edge_case(self, df):
        """Network-bound scenario: very high net_rx but moderate cpu_mean."""
        mask = (df["net_rx_mean"] > 50e6) & (df["cpu_mean"] < 0.50)
        assert mask.any(), "No network-bound edge-case rows found"

    def test_fully_saturated_edge_case(self, df):
        """Fully-saturated scenario: cpu_mean > 0.90, mem_mean > 0.90."""
        mask = (df["cpu_mean"] > 0.90) & (df["mem_mean"] > 0.90)
        assert mask.any(), "No fully-saturated edge-case rows found"

    # --- Determinism ---

    def test_deterministic_with_same_seed(self):
        """Two calls with the same seed must produce identical DataFrames."""
        df1 = train_model.generate_dataset(n_samples=200)
        df2 = train_model.generate_dataset(n_samples=200)
        assert df1.equals(df2)


# ===========================================================================
# Section 2 — feature_store.py / NodeMetricsWindow
# ===========================================================================

class TestNodeMetricsWindow:
    """Unit tests for the per-node sliding window and feature aggregation."""

    def _window(self, size=WINDOW_SIZE) -> NodeMetricsWindow:
        return NodeMetricsWindow(size)

    # --- Empty window ---

    def test_empty_feature_dict_has_all_keys(self):
        w = self._window()
        d = w.to_feature_dict()
        for col in FEATURE_COLUMNS:
            assert col in d, f"Missing key '{col}' in empty feature dict"

    def test_empty_feature_dict_all_zeros(self):
        w = self._window()
        d = w.to_feature_dict()
        for col in FEATURE_COLUMNS:
            assert d[col] == 0.0, f"Expected 0.0 for '{col}', got {d[col]}"

    # --- Single push ---

    def test_single_push_mean_equals_value(self):
        w = self._window()
        w.push(cpu=0.5, mem=0.4, net_rx=1000.0, net_tx=2000.0, load=1.5)
        d = w.to_feature_dict()
        assert math.isclose(d["cpu_mean"], 0.5, rel_tol=1e-9)
        assert math.isclose(d["mem_mean"], 0.4, rel_tol=1e-9)
        assert math.isclose(d["net_rx_mean"], 1000.0, rel_tol=1e-9)
        assert math.isclose(d["net_tx_mean"], 2000.0, rel_tol=1e-9)
        assert math.isclose(d["load_mean"], 1.5, rel_tol=1e-9)

    def test_single_push_max_equals_mean(self):
        w = self._window()
        w.push(cpu=0.3, mem=0.6, net_rx=500.0, net_tx=500.0, load=0.8)
        d = w.to_feature_dict()
        assert math.isclose(d["cpu_max"], d["cpu_mean"])
        assert math.isclose(d["mem_max"], d["mem_mean"])
        assert math.isclose(d["load_max"], d["load_mean"])

    def test_single_push_std_is_zero(self):
        w = self._window()
        w.push(cpu=0.2, mem=0.3, net_rx=0.0, net_tx=0.0, load=0.5)
        d = w.to_feature_dict()
        assert math.isclose(d["cpu_std"], 0.0, abs_tol=1e-12)
        assert math.isclose(d["mem_std"], 0.0, abs_tol=1e-12)

    # --- Multi-push aggregation ---

    def test_mean_is_correct(self):
        w = self._window()
        values = [0.2, 0.4, 0.6]
        for v in values:
            w.push(cpu=v, mem=0.0, net_rx=0.0, net_tx=0.0, load=0.0)
        d = w.to_feature_dict()
        expected_mean = sum(values) / len(values)
        assert math.isclose(d["cpu_mean"], expected_mean, rel_tol=1e-9)

    def test_max_is_correct(self):
        w = self._window()
        for v in [0.1, 0.9, 0.5]:
            w.push(cpu=v, mem=0.0, net_rx=0.0, net_tx=0.0, load=0.0)
        d = w.to_feature_dict()
        assert math.isclose(d["cpu_max"], 0.9, rel_tol=1e-9)

    def test_std_is_correct(self):
        w = self._window()
        vals = [0.0, 0.5, 1.0]
        for v in vals:
            w.push(cpu=v, mem=0.0, net_rx=0.0, net_tx=0.0, load=0.0)
        d = w.to_feature_dict()
        # Population std (not sample std, matching the implementation)
        mean = sum(vals) / len(vals)
        expected_std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        assert math.isclose(d["cpu_std"], expected_std, rel_tol=1e-9)

    # --- Spike rate ---

    def test_spike_rate_zero_when_all_below_threshold(self):
        w = self._window()
        for v in [0.1, 0.2, 0.3]:  # all < 0.70
            w.push(cpu=v, mem=0.0, net_rx=0.0, net_tx=0.0, load=0.0)
        d = w.to_feature_dict()
        assert d["cpu_spike_rate"] == 0.0

    def test_spike_rate_one_when_all_above_threshold(self):
        w = self._window()
        for v in [0.8, 0.9, 0.95]:  # all > 0.70
            w.push(cpu=v, mem=0.0, net_rx=0.0, net_tx=0.0, load=0.0)
        d = w.to_feature_dict()
        assert math.isclose(d["cpu_spike_rate"], 1.0, rel_tol=1e-9)

    def test_spike_rate_partial(self):
        w = self._window()
        # 2 spikes out of 4 readings
        for v in [0.9, 0.1, 0.8, 0.2]:
            w.push(cpu=v, mem=0.0, net_rx=0.0, net_tx=0.0, load=0.0)
        d = w.to_feature_dict()
        assert math.isclose(d["cpu_spike_rate"], 0.5, rel_tol=1e-9)

    def test_spike_rate_boundary_at_threshold(self):
        """CPU exactly at 0.70 should NOT count as a spike (> not >=)."""
        w = self._window()
        w.push(cpu=0.70, mem=0.0, net_rx=0.0, net_tx=0.0, load=0.0)
        d = w.to_feature_dict()
        assert d["cpu_spike_rate"] == 0.0

    # --- Sliding window eviction ---

    def test_sliding_window_evicts_oldest(self):
        """When window is full, oldest reading should be evicted."""
        size = 3
        w = self._window(size)
        for v in [0.1, 0.2, 0.3]:
            w.push(cpu=v, mem=0.0, net_rx=0.0, net_tx=0.0, load=0.0)
        # Push a 4th value; 0.1 should be evicted
        w.push(cpu=0.9, mem=0.0, net_rx=0.0, net_tx=0.0, load=0.0)
        d = w.to_feature_dict()
        # Window now: [0.2, 0.3, 0.9] → mean = 0.467, max = 0.9
        assert math.isclose(d["cpu_max"], 0.9, rel_tol=1e-9)
        expected_mean = (0.2 + 0.3 + 0.9) / 3
        assert math.isclose(d["cpu_mean"], expected_mean, rel_tol=1e-9)

    def test_window_size_respected(self):
        """Pushing more than window_size readings never grows the internal deque."""
        size = 4
        w = self._window(size)
        for i in range(20):
            w.push(cpu=float(i) / 20, mem=0.0, net_rx=0.0, net_tx=0.0, load=0.0)
        assert len(w.cpu_util) == size

    # --- Feature dict key ordering contract ---

    def test_feature_dict_keys_match_feature_columns(self):
        """Keys returned by to_feature_dict must exactly match FEATURE_COLUMNS."""
        w = self._window()
        w.push(cpu=0.5, mem=0.5, net_rx=1.0, net_tx=1.0, load=1.0)
        d = w.to_feature_dict()
        assert list(d.keys()) == FEATURE_COLUMNS


class TestFeatureStore:
    """FeatureStore public API behaviour without live Prometheus/K8s."""

    def _store(self) -> FeatureStore:
        return FeatureStore()

    def test_get_features_unknown_node_returns_all_zeros(self):
        store = self._store()
        d = store.get_features("nonexistent-node")
        for col in FEATURE_COLUMNS:
            assert d[col] == 0.0

    def test_get_features_unknown_node_has_all_keys(self):
        store = self._store()
        d = store.get_features("nonexistent-node")
        assert set(d.keys()) == set(FEATURE_COLUMNS)

    def test_known_nodes_empty_initially(self):
        store = self._store()
        assert store.known_nodes() == []

    def test_known_nodes_after_internal_push(self):
        """Simulate a push by directly writing into _windows."""
        store = self._store()
        w = NodeMetricsWindow(WINDOW_SIZE)
        w.push(cpu=0.3, mem=0.4, net_rx=0.0, net_tx=0.0, load=0.5)
        with store._lock:
            store._windows["node-a"] = w
        assert "node-a" in store.known_nodes()

    def test_get_features_returns_pushed_data(self):
        store = self._store()
        w = NodeMetricsWindow(WINDOW_SIZE)
        w.push(cpu=0.7, mem=0.6, net_rx=100.0, net_tx=200.0, load=2.0)
        with store._lock:
            store._windows["node-b"] = w
        d = store.get_features("node-b")
        assert math.isclose(d["cpu_mean"], 0.7, rel_tol=1e-9)
        assert math.isclose(d["mem_mean"], 0.6, rel_tol=1e-9)

    def test_parse_cpu_millicores(self):
        """'450m' should parse to 0.45 cores."""
        assert math.isclose(FeatureStore._parse_cpu("450m"), 0.45)

    def test_parse_cpu_whole_cores(self):
        assert math.isclose(FeatureStore._parse_cpu("2"), 2.0)

    def test_parse_cpu_fractional_string(self):
        assert math.isclose(FeatureStore._parse_cpu("1.5"), 1.5)

    def test_get_features_is_thread_safe(self):
        """Concurrent reads from multiple threads should not raise."""
        store = self._store()
        w = NodeMetricsWindow(WINDOW_SIZE)
        for _ in range(WINDOW_SIZE):
            w.push(cpu=0.5, mem=0.5, net_rx=0.0, net_tx=0.0, load=1.0)
        with store._lock:
            store._windows["node-c"] = w

        errors = []
        def reader():
            try:
                for _ in range(50):
                    store.get_features("node-c")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread-safety errors: {errors}"


# ===========================================================================
# Section 3 — model.py / NodeScorer
# ===========================================================================

def _healthy_features() -> dict:
    """Feature dict representing a near-idle node — should score close to 100."""
    return {col: 0.0 for col in FEATURE_COLUMNS}


def _stressed_features() -> dict:
    """Feature dict representing a fully-saturated node — should score 0."""
    return {
        "cpu_mean":       1.0,
        "cpu_max":        1.0,
        "cpu_std":        0.30,   # max burst std — drives std_score to 0
        "cpu_spike_rate": 1.0,
        "mem_mean":       1.0,
        "mem_max":        1.0,
        "mem_std":        0.10,
        "load_mean":      16.0,
        "load_max":       20.0,
        "net_rx_mean":    500e6,  # 500 MB/s rx — drives net_score to 0
        "net_tx_mean":    500e6,  # 500 MB/s tx
    }


class TestNodeScorerHeuristic:
    """NodeScorer heuristic fallback (no model file required)."""

    @pytest.fixture
    def scorer(self):
        s = NodeScorer()
        # Do NOT call s.load() — forces heuristic path
        return s

    def test_source_is_heuristic_before_load(self, scorer):
        assert scorer.source == "heuristic"

    def test_predict_returns_tuple(self, scorer):
        result = scorer.predict(_healthy_features())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_returns_int_score(self, scorer):
        score, _ = scorer.predict(_healthy_features())
        assert isinstance(score, int)

    def test_predict_returns_source_string(self, scorer):
        _, source = scorer.predict(_healthy_features())
        assert isinstance(source, str)

    def test_heuristic_source_label(self, scorer):
        _, source = scorer.predict(_healthy_features())
        assert source == "heuristic"

    def test_healthy_node_scores_high(self, scorer):
        """All-zero features → score should be 100 (nothing using resources)."""
        score, _ = scorer.predict(_healthy_features())
        assert score == 100

    def test_stressed_node_scores_low(self, scorer):
        """All-maxed features → score should be 0."""
        score, _ = scorer.predict(_stressed_features())
        assert score == 0

    def test_score_is_clamped_below_zero(self, scorer):
        """Extreme high values must produce a score >= 0."""
        # All metrics at max -> heuristic composite = 0 -> score = 0
        feats = {col: 1.0 for col in FEATURE_COLUMNS}
        feats["load_mean"] = 100.0
        score, _ = scorer.predict(feats)
        assert score >= 0

    def test_heuristic_intermediate_node_score_in_range(self, scorer):
        """Heuristic output for any realistic inputs must stay in [0, 100]."""
        # The heuristic formula: each sub-score is clamped to [0,1] before summing,
        # so composite is always in [0,1] and the final int is always in [0,100].
        feats = {
            "cpu_mean":       0.5,
            "cpu_max":        0.6,
            "cpu_std":        0.05,
            "cpu_spike_rate": 0.3,
            "mem_mean":       0.5,
            "mem_max":        0.6,
            "mem_std":        0.03,
            "load_mean":      2.0,
            "load_max":       3.0,
            "net_rx_mean":    5e6,
            "net_tx_mean":    5e6,
        }
        score, _ = scorer.predict(feats)
        assert 0 <= score <= 100

    def test_partial_stress_score_intermediate(self, scorer):
        """A moderately loaded node should score between 0 and 100 exclusively."""
        feats = dict(_healthy_features())
        feats["cpu_mean"] = 0.5
        feats["mem_mean"] = 0.5
        feats["cpu_spike_rate"] = 0.2
        feats["load_mean"] = 2.0
        score, _ = scorer.predict(feats)
        assert 0 < score < 100

    def test_missing_keys_default_to_zero(self, scorer):
        """predict() must not raise when optional keys are absent."""
        score, _ = scorer.predict({})  # empty dict
        assert score == 100  # all defaults = 0.0 → full score

    def test_feature_vector_ordering_consistency(self, scorer):
        """Score must be the same regardless of dict insertion order."""
        feats_a = {col: 0.3 for col in FEATURE_COLUMNS}
        feats_b = {col: 0.3 for col in reversed(FEATURE_COLUMNS)}
        score_a, _ = scorer.predict(feats_a)
        score_b, _ = scorer.predict(feats_b)
        assert score_a == score_b

    # --- Edge cases ---

    def test_cpu_spike_only_penalises_score(self, scorer):
        """Raising cpu_spike_rate alone should lower the score."""
        low_spike = dict(_healthy_features())
        low_spike["cpu_spike_rate"] = 0.0
        high_spike = dict(_healthy_features())
        high_spike["cpu_spike_rate"] = 1.0
        score_low, _ = scorer.predict(low_spike)
        score_high, _ = scorer.predict(high_spike)
        assert score_low > score_high

    def test_load_normalisation_clips_at_100_percent(self, scorer):
        """Load far above 4× should not crash; score should remain >= 0."""
        feats_high_load = dict(_healthy_features())
        feats_high_load["load_mean"] = 100.0   # extreme value
        score, _ = scorer.predict(feats_high_load)
        # load contributes 12%; the other 88% stays at 1.0 -> score = 88
        assert score >= 0

    def test_net_saturation_penalises_score(self, scorer):
        """High combined rx+tx should lower the score vs. a quiet node."""
        quiet_net = dict(_healthy_features())  # net_rx=net_tx=0 -> net_score=1.0
        busy_net  = dict(_healthy_features())
        busy_net["net_rx_mean"] = 500e6  # 500 MB/s rx
        busy_net["net_tx_mean"] = 500e6  # 500 MB/s tx -> combined at NIC cap -> net_score=0
        score_quiet, _ = scorer.predict(quiet_net)
        score_busy,  _ = scorer.predict(busy_net)
        assert score_quiet > score_busy, (
            f"Expected quiet net ({score_quiet}) > saturated net ({score_busy})"
        )

    def test_cpu_std_burst_penalises_score(self, scorer):
        """High cpu_std (volatile/flapping) should lower the score vs. stable node."""
        stable  = dict(_healthy_features())   # cpu_std=0 -> std_score=1.0
        bursty  = dict(_healthy_features())
        bursty["cpu_std"] = 0.30              # max training std -> std_score=0
        score_stable, _ = scorer.predict(stable)
        score_bursty, _ = scorer.predict(bursty)
        assert score_stable > score_bursty, (
            f"Expected stable std ({score_stable}) > bursty std ({score_bursty})"
        )


class TestNodeScorerXGBoost:
    """
    NodeScorer XGBoost path.

    Strategy: instead of pickling a MagicMock (which fails because pickle
    cannot serialize dynamically-created mock classes), we inject the mock
    model directly into scorer._model and set scorer._source manually so
    that the load() path is bypassed but the predict() routing is exercised.
    """

    def _make_scorer_with_mock(self, predict_return) -> tuple:
        """Return (scorer, mock_model) with _model pre-injected."""
        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.array([predict_return])
        scorer = NodeScorer()
        scorer._model = mock_model
        scorer._source = "xgboost"
        return scorer, mock_model

    def test_xgboost_path_used_when_model_injected(self):
        """With _model set, source must report 'xgboost'."""
        scorer, _ = self._make_scorer_with_mock(72.5)
        assert scorer.source == "xgboost"

    def test_xgboost_predict_returns_xgboost_source_label(self):
        scorer, _ = self._make_scorer_with_mock(55.0)
        _, source = scorer.predict(_healthy_features())
        assert source == "xgboost"

    def test_xgboost_score_clamped_and_rounded_above_max(self):
        """XGBoost raw output > 100 should be clamped to 100."""
        scorer, _ = self._make_scorer_with_mock(101.9)
        score, source = scorer.predict(_healthy_features())
        assert score == 100
        assert source == "xgboost"

    def test_xgboost_negative_clamped_to_zero(self):
        """XGBoost raw output < 0 should be clamped to 0."""
        scorer, _ = self._make_scorer_with_mock(-5.0)
        score, _ = scorer.predict(_healthy_features())
        assert score == 0

    def test_xgboost_midrange_score_returned_as_int(self):
        scorer, _ = self._make_scorer_with_mock(63.7)
        score, _ = scorer.predict(_healthy_features())
        assert isinstance(score, int)
        assert score == 64   # round(63.7)

    def test_corrupt_model_file_falls_back_to_heuristic(self, tmp_path):
        """A corrupted pickle must not crash; scorer falls back to heuristic."""
        model_path = tmp_path / "niyojak_model.pkl"
        model_path.write_bytes(b"this is not a pickle")

        scorer = NodeScorer()
        with mock.patch.object(model_module, "MODEL_PATH", str(model_path)):
            scorer.load()

        assert scorer.source == "heuristic"

    def test_missing_model_file_falls_back_to_heuristic(self, tmp_path):
        """A missing model file must not crash; scorer falls back to heuristic."""
        absent_path = str(tmp_path / "absent.pkl")
        scorer = NodeScorer()
        with mock.patch.object(model_module, "MODEL_PATH", absent_path):
            scorer.load()
        assert scorer.source == "heuristic"

    def test_xgboost_receives_correct_feature_vector_length(self):
        """The feature vector passed to model.predict() must have exactly 11 elements."""
        captured = {}
        mock_model = mock.MagicMock()

        def capture_predict(vec):
            captured["vec"] = vec
            return np.array([50.0])

        mock_model.predict.side_effect = capture_predict
        scorer = NodeScorer()
        scorer._model = mock_model
        scorer._source = "xgboost"

        scorer.predict(_healthy_features())
        assert captured["vec"].shape == (1, len(FEATURE_COLUMNS))  # (1, 11)


# ===========================================================================
# Section 4 — main.py / FastAPI /score endpoint
# ===========================================================================

# Import starlette test client lazily so the module can be loaded even if
# fastapi / httpx are not installed (though they are in requirements.txt).
try:
    from fastapi.testclient import TestClient
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

pytestmark_fastapi = pytest.mark.skipif(
    not _FASTAPI_AVAILABLE,
    reason="fastapi[testclient] not installed",
)


@pytest.fixture(scope="module")
def client():
    """
    Return a FastAPI TestClient with the lifespan startup bypassed.

    We patch FeatureStore.start() and NodeScorer.load() to prevent real
    Prometheus / K8s / filesystem calls during testing.
    """
    if not _FASTAPI_AVAILABLE:
        pytest.skip("fastapi not available")

    with (
        mock.patch("feature_store.FeatureStore.start"),
        mock.patch("model.NodeScorer.load"),
    ):
        import main as main_module  # type: ignore[import-not-found]
        # TestClient accepts the app as the first positional argument.
        # Using it as a context manager triggers lifespan startup/shutdown.
        with TestClient(main_module.app) as c:
            yield c


class TestScoreEndpoint:
    """POST /score — request handling, response schema, and score range."""

    def _payload(self, **overrides) -> dict:
        base = {
            "pod_name": "test-pod",
            "pod_namespace": "default",
            "node_name": "node-1",
        }
        base.update(overrides)
        return base

    @pytestmark_fastapi
    def test_score_returns_200(self, client):
        resp = client.post("/score", json=self._payload())
        assert resp.status_code == 200

    @pytestmark_fastapi
    def test_score_response_has_score_field(self, client):
        resp = client.post("/score", json=self._payload())
        assert "score" in resp.json()

    @pytestmark_fastapi
    def test_score_response_has_reason_field(self, client):
        resp = client.post("/score", json=self._payload())
        assert "reason" in resp.json()

    @pytestmark_fastapi
    def test_score_response_has_source_field(self, client):
        resp = client.post("/score", json=self._payload())
        assert "source" in resp.json()

    @pytestmark_fastapi
    def test_score_is_integer(self, client):
        resp = client.post("/score", json=self._payload())
        score = resp.json()["score"]
        assert isinstance(score, int)

    @pytestmark_fastapi
    def test_score_within_valid_range(self, client):
        resp = client.post("/score", json=self._payload())
        score = resp.json()["score"]
        assert 0 <= score <= 100

    @pytestmark_fastapi
    def test_score_accepts_optional_fields(self, client):
        """Optional pod_labels, cpu_request_millicores, memory_request_bytes."""
        payload = self._payload(
            pod_labels={"app": "web"},
            cpu_request_millicores=250,
            memory_request_bytes=134217728,
        )
        resp = client.post("/score", json=payload)
        assert resp.status_code == 200

    @pytestmark_fastapi
    def test_score_missing_required_field_returns_422(self, client):
        """Missing node_name should return 422 Unprocessable Entity."""
        resp = client.post("/score", json={"pod_name": "x", "pod_namespace": "y"})
        assert resp.status_code == 422

    @pytestmark_fastapi
    def test_source_is_heuristic_or_xgboost(self, client):
        resp = client.post("/score", json=self._payload())
        source = resp.json()["source"]
        assert source in ("heuristic", "xgboost")

    @pytestmark_fastapi
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert body["status"] == "ok"
