import pytest

from data.rtptorrent import (
    attach_changed_files,
    load_changed_files_csv,
    load_project_csv,
    load_rtptorrent_v1_project,
)


def test_load_project_csv_groups_rows_into_sorted_cycles(tmp_path):
    csv_path = tmp_path / "okhttp.csv"
    csv_path.write_text(
        "\n".join(
            [
                "job_id,commit_id,test,failures,errors,skipped,duration",
                "2,def,TestB,1,0,0,0.2",
                "1,abc,TestA,0,0,0,0.1",
                "2,def,TestC,0,1,0,0.3",
            ]
        )
        + "\n"
    )

    cycles = load_project_csv(csv_path)

    assert [cycle.job_id for cycle in cycles] == ["1", "2"]
    assert cycles[0].commit_id == "abc"
    assert len(cycles[1].outcomes) == 2
    assert {outcome.test_name for outcome in cycles[1].outcomes} == {"TestB", "TestC"}
    assert all(outcome.failed for outcome in cycles[1].outcomes)


def test_load_project_csv_accepts_column_aliases_and_limit(tmp_path):
    csv_path = tmp_path / "sling.csv"
    csv_path.write_text(
        "\n".join(
            [
                "tr_job_id,sha,testcase,failed,error,skip,time",
                "job-b,bbb,TestB,0,0,1,0",
                "job-a,aaa,TestA,0,0,0,1.5",
            ]
        )
        + "\n"
    )

    cycles = load_project_csv(csv_path, max_cycles=1)

    assert len(cycles) == 1
    assert cycles[0].job_id == "job-a"
    assert cycles[0].outcomes[0].duration == 1.5
    assert not cycles[0].outcomes[0].failed


def test_load_project_csv_joins_changed_files_by_commit(tmp_path):
    project_csv = tmp_path / "okhttp.csv"
    project_csv.write_text(
        "\n".join(
            [
                "job_id,commit_id,test,failures",
                "1,abc,TestA,0",
                "2,def,TestB,1",
            ]
        )
        + "\n"
    )
    changes_csv = tmp_path / "changes.csv"
    changes_csv.write_text(
        "\n".join(
            [
                "commit_id,file_path",
                "abc,src/A.java",
                "def,src/B.java",
                "def,src/C.java",
            ]
        )
        + "\n"
    )

    cycles = load_project_csv(project_csv, changes_csv=changes_csv)

    assert cycles[0].changed_files == frozenset({"src/A.java"})
    assert cycles[1].changed_files == frozenset({"src/B.java", "src/C.java"})


def test_load_rtptorrent_v1_schema_joins_commits_and_patches(tmp_path):
    project_csv = tmp_path / "apache@sling.csv"
    built_csv = tmp_path / "tr_all_built_commits.csv"
    patches_csv = tmp_path / "apache@sling-patches.csv"
    project_csv.write_text(
        "\n".join(
            [
                "travisJobId,testName,index,duration,count,failures,errors,skipped",
                "3,TestA,0,0.1,1,0,0,0",
                "1,TestB,0,0.1,1,1,0,0",
                "2,TestC,0,0.1,1,0,1,0",
            ]
        )
        + "\n"
    )
    built_csv.write_text(
        "\n".join(
            [
                "tr_job_id,git_commit_id",
                "1,c1",
                "2,c2",
                "2,c3",
                "3,c4",
            ]
        )
        + "\n"
    )
    patches_csv.write_text(
        "\n".join(
            [
                "sha,name",
                "c1,src/A.java",
                "c2,src/B.java",
                "c3,src/C.java",
                *[f"c4,infra/{idx}.yml" for idx in range(31)],
            ]
        )
        + "\n"
    )

    cycles, stats = load_rtptorrent_v1_project(
        project_csv,
        built_csv,
        patches_csv,
        max_changed_files=30,
    )

    assert [cycle.job_id for cycle in cycles] == ["1", "2"]
    assert cycles[0].changed_files == frozenset({"src/A.java"})
    assert cycles[1].changed_files == frozenset({"src/B.java", "src/C.java"})
    assert cycles[0].outcomes[0].test_name == "TestB"
    assert cycles[0].outcomes[0].failed
    assert cycles[1].outcomes[0].failed
    assert stats.raw_jobs == 3
    assert stats.dropped_large_change_jobs == 1


def test_attach_changed_files_prefers_job_specific_metadata(tmp_path):
    project_csv = tmp_path / "sling.csv"
    project_csv.write_text(
        "\n".join(
            [
                "job_id,commit_id,test,failures",
                "1,same,TestA,0",
                "2,same,TestB,0",
            ]
        )
        + "\n"
    )
    changes_csv = tmp_path / "changes.csv"
    changes_csv.write_text(
        "\n".join(
            [
                "job_id,commit_id,files",
                "1,same,src/A.java;src/B.java",
                "2,same,src/C.java",
            ]
        )
        + "\n"
    )

    cycles = attach_changed_files(load_project_csv(project_csv), changes_csv)

    assert cycles[0].changed_files == frozenset({"src/A.java", "src/B.java"})
    assert cycles[1].changed_files == frozenset({"src/C.java"})


def test_load_changed_files_csv_accepts_delimited_file_lists(tmp_path):
    changes_csv = tmp_path / "changes.csv"
    changes_csv.write_text("sha,changed_files\nabc,src/A.java|src/B.py\n")

    by_job, by_commit = load_changed_files_csv(changes_csv)

    assert by_job == {}
    assert by_commit["abc"] == frozenset({"src/A.java", "src/B.py"})


def test_load_project_csv_requires_job_and_test_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("job_id,failures\n1,0\n")

    with pytest.raises(ValueError, match="test"):
        load_project_csv(csv_path)
