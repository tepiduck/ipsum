from experiments import artifacts
from experiments.compounding import (
    DataMatchedControl,
    RTPTorrentCompoundingConfig,
    SynthCompoundingConfig,
    compounding_curve,
    run_rtptorrent,
    run_synth,
    slope_gap,
)
from experiments.compounding_rtptorrent import (
    RTPTorrentRealDataConfig,
    _coarsen_changed_files,
    run as run_rtptorrent_real,
)
from experiments.compounding import _synth_timeline
from experiments.instrument_self_check import (
    InstrumentSelfCheckConfig,
    _run_mode,
    score_test_recall_at_selection_rate,
)
from experiments.compounding import Cycle, SelectionCommit, SelectionOutcome


def test_compounding_curve_scores_before_observe():
    config = SynthCompoundingConfig(
        n_files=24,
        n_tests=12,
        n_clusters=4,
        cycles=60,
        eval_interval=20,
        eval_window=20,
        seed=11,
    )
    timeline = _synth_timeline(config)
    system = DataMatchedControl(config.n_tests, config.selection_rate_cap)

    curve = compounding_curve(system, timeline, eval_interval=20, eval_window=20)

    assert len(curve) == 2
    assert all(0.0 <= point.test_recall <= 1.0 for point in curve)
    assert all(point.selection_rate <= 0.34 for point in curve)
    assert len(system.examples) == config.cycles * config.n_tests


def test_slope_gap_positive_when_ipsum_curve_improves_more():
    baseline = [
        _point(10, 0.2),
        _point(20, 0.2),
        _point(30, 0.2),
    ]
    ipsum = [
        _point(10, 0.2),
        _point(20, 0.4),
        _point(30, 0.6),
    ]

    assert slope_gap(ipsum, baseline) > 0.0


def test_score_test_recall_at_selection_rate_hand_built_case():
    cycles = [
        Cycle(
            cycle=0,
            commit=SelectionCommit(cycle=0, changed_files=frozenset({1})),
            outcomes=(
                SelectionOutcome(test=0, failed=True),
                SelectionOutcome(test=1, failed=False),
                SelectionOutcome(test=2, failed=True),
            ),
        ),
        Cycle(
            cycle=1,
            commit=SelectionCommit(cycle=1, changed_files=frozenset({2})),
            outcomes=(
                SelectionOutcome(test=0, failed=False),
                SelectionOutcome(test=1, failed=True),
                SelectionOutcome(test=2, failed=False),
            ),
        ),
    ]
    selections = [{0}, {2}]

    point = score_test_recall_at_selection_rate(cycles, selections, n_tests=3)

    assert point.test_recall == 1 / 3
    assert point.selection_rate == 2 / 6


def test_instrument_negative_control_has_near_zero_gap():
    config = InstrumentSelfCheckConfig(
        n_files=32,
        n_tests=16,
        n_clusters=4,
        cycles=160,
        drift_schedule=(80, 120),
        eval_interval=40,
        eval_window=40,
        admission_warmup=60,
        admission_interval=40,
        validation_cycles=30,
        seed=22,
    )

    result = _run_mode(config, mode="negative")

    assert abs(result["metrics"]["ipsum_vs_data_matched_slope_gap"]) <= 1e-12
    assert abs(result["metrics"]["final_gap"]) <= 1e-12
    assert abs(result["metrics"]["half_gap_delta"]) <= 1e-12
    assert result["metrics"]["negative_control_passed"] == 1.0


def test_run_synth_emits_three_system_slope_series(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RUNS_DIR", tmp_path)
    config = SynthCompoundingConfig(
        n_files=32,
        n_tests=16,
        n_clusters=4,
        cycles=160,
        eval_interval=40,
        eval_window=40,
        admission_warmup=60,
        admission_interval=40,
        validation_cycles=30,
        seed=12,
    )

    result = run_synth(config)

    assert result["metrics"]["admitted_abstractions_final"] >= 0.0
    slope_path = tmp_path / result["run_id"] / "slope.json"
    slope_text = slope_path.read_text()
    assert "weekly_retrain" in slope_text
    assert "data_matched_control" in slope_text
    assert "ipsum" in slope_text


def test_run_rtptorrent_emits_three_system_slope_series(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RUNS_DIR", tmp_path / "runs")
    project_csv = tmp_path / "okhttp.csv"
    changes_csv = tmp_path / "changes.csv"
    project_rows = ["job_id,commit_id,test,failures"]
    change_rows = ["commit_id,changed_files"]
    for cycle in range(1, 81):
        commit = f"c{cycle}"
        hot_file = "src/Auth.java" if cycle <= 40 else "src/Billing.java"
        change_rows.append(f"{commit},{hot_file}")
        project_rows.extend(
            [
                f"{cycle},{commit},AuthTest,{int(cycle <= 40 and cycle % 2 == 0)}",
                f"{cycle},{commit},BillingTest,{int(cycle > 40 and cycle % 2 == 0)}",
                f"{cycle},{commit},SmokeTest,0",
                f"{cycle},{commit},ParserTest,{int(cycle % 17 == 0)}",
            ]
        )
    project_csv.write_text("\n".join(project_rows) + "\n")
    changes_csv.write_text("\n".join(change_rows) + "\n")
    config = RTPTorrentCompoundingConfig(
        project_csv=str(project_csv),
        changes_csv=str(changes_csv),
        eval_interval=20,
        eval_window=20,
        admission_warmup=20,
        admission_interval=20,
        validation_cycles=10,
        weekly_retrain_interval=20,
        weekly_retrain_window=40,
        min_support=2,
        cochange_threshold=0.5,
    )

    result = run_rtptorrent(config)

    assert result["metrics"]["changed_file_coverage"] == 1.0
    slope_text = (tmp_path / "runs" / result["run_id"] / "slope.json").read_text()
    assert "weekly_retrain" in slope_text
    assert "data_matched_control" in slope_text
    assert "ipsum" in slope_text


def test_real_rtptorrent_run_joins_v1_schema_and_emits_compounding(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RUNS_DIR", tmp_path / "runs")
    project_rows = ["job_id,commit_id,test,failures"]
    built_rows = ["tr_job_id,git_commit_id"]
    patch_rows = ["sha,name"]
    for cycle in range(1, 61):
        commit = f"c{cycle}"
        changed_file = "src/Auth.java" if cycle <= 30 else "src/Billing.java"
        built_rows.append(f"{cycle},{commit}")
        patch_rows.append(f"{commit},{changed_file}")
        project_rows.extend(
            [
                f"{cycle},{commit},AuthTest,{int(cycle <= 30 and cycle % 3 == 0)}",
                f"{cycle},{commit},BillingTest,{int(cycle > 30 and cycle % 3 == 0)}",
                f"{cycle},{commit},SmokeTest,0",
            ]
        )
    project_csv = tmp_path / "okhttp.csv"
    built_csv = tmp_path / "tr_all_built_commits.csv"
    patches_csv = tmp_path / "okhttp-patches.csv"
    project_csv.write_text("\n".join(project_rows) + "\n")
    built_csv.write_text("\n".join(built_rows) + "\n")
    patches_csv.write_text("\n".join(patch_rows) + "\n")

    config = RTPTorrentRealDataConfig(
        dataset="okhttp",
        project_csv=str(project_csv),
        built_commits_csv=str(built_csv),
        patches_csv=str(patches_csv),
        eval_interval=20,
        eval_window=20,
        admission_warmup=20,
        admission_interval=20,
        validation_cycles=10,
        adaptation_window=30,
        weekly_retrain_interval=20,
        weekly_retrain_window=40,
        change_path_depth=1,
        min_support=2,
        cochange_threshold=0.5,
    )

    result = run_rtptorrent_real(config)

    assert result["metrics"]["used_cycles"] == 60.0
    assert result["metrics"]["changed_file_coverage"] > 0.9
    assert result["metrics"]["distinct_change_tokens"] == 1.0
    assert result["metrics"]["admission_funnel_cycles"] > 0.0
    assert result["metrics"]["candidates_proposed_total"] >= 0.0
    slope_text = (tmp_path / "runs" / result["run_id"] / "slope.json").read_text()
    meta_text = (tmp_path / "runs" / result["run_id"] / "meta.json").read_text()
    assert '"card": "compounding"' in meta_text
    assert "weekly_retrain" in slope_text
    assert "data_matched_control" in slope_text
    assert "ipsum" in slope_text


def test_rtptorrent_change_coarsening_supports_directory_and_java_package():
    files = frozenset(
        {
            "src/main/java/com/squareup/okhttp/Call.java",
            "src/test/java/com/squareup/okhttp/CallTest.java",
            "okhttp/src/main/java/okhttp3/internal/http/Retry.java",
        }
    )

    assert _coarsen_changed_files(files, granularity="directory", depth=3) == frozenset(
        {"src/main/java", "src/test/java", "okhttp/src/main"}
    )
    assert _coarsen_changed_files(files, granularity="java_package", depth=3) == frozenset(
        {"com.squareup.okhttp", "okhttp3.internal.http"}
    )


def _point(cycle: int, recall: float):
    from experiments.compounding import Point

    return Point(
        cycle=cycle,
        repo_age_days=float(cycle),
        test_recall=recall,
        selection_rate=0.33,
    )
