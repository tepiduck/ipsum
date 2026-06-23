from data.rtptorrent import RTPTorrentCycle, RTPTorrentOutcome
from experiments.baseline_weekly_retrain import WeeklyRetrainBaseline


def test_weekly_baseline_extracts_change_test_features():
    baseline = WeeklyRetrainBaseline(selection_rate_cap=0.5)
    baseline.observe(
        _cycle(
            1,
            ["src/Auth.java"],
            [
                ("AuthTest", True),
                ("BillingTest", False),
            ],
        )
    )
    baseline.retrain()

    features = baseline.features(_cycle(2, ["src/Auth.java"], []), "AuthTest")

    assert features["changed_file_count"] == 1.0
    assert features["java_file_fraction"] == 1.0
    assert features["historical_failure_rate"] > 0.0
    assert features["related_file_failure_rate"] > 0.0


def test_weekly_baseline_selects_highest_scored_tests_after_retrain():
    baseline = WeeklyRetrainBaseline(selection_rate_cap=0.5)
    baseline.observe(
        _cycle(
            1,
            ["src/Auth.java"],
            [
                ("AuthTest", True),
                ("BillingTest", False),
            ],
        )
    )
    baseline.observe(
        _cycle(
            2,
            ["src/Auth.java"],
            [
                ("AuthTest", True),
                ("BillingTest", False),
            ],
        )
    )
    baseline.retrain()

    selected = baseline.select(
        _cycle(
            3,
            ["src/Auth.java"],
            [
                ("AuthTest", False),
                ("BillingTest", False),
            ],
        )
    )

    assert selected == {"AuthTest"}


def test_weekly_baseline_retrains_on_interval():
    baseline = WeeklyRetrainBaseline(selection_rate_cap=1.0, retrain_interval=2)

    baseline.observe(_cycle(1, ["src/A.java"], [("ATest", True)]))
    assert baseline._model == {}

    baseline.observe(_cycle(2, ["src/A.java"], [("ATest", True)]))
    assert baseline._model["ATest"] > 0.5


def _cycle(cycle: int, files: list[str], outcomes: list[tuple[str, bool]]) -> RTPTorrentCycle:
    return RTPTorrentCycle(
        cycle=cycle,
        job_id=str(cycle),
        commit_id=f"commit-{cycle}",
        changed_files=frozenset(files),
        outcomes=tuple(
            RTPTorrentOutcome(
                test_name=name,
                failed=failed,
                failures=int(failed),
                errors=0,
                skipped=0,
                duration=0.1,
            )
            for name, failed in outcomes
        ),
    )
