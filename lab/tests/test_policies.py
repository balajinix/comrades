"""Tests for the scripted modality policies."""
from __future__ import annotations

from dal_lab.policies import plan_rows
from dal_lab.workload import WORKFLOWS, load_questions, same_answer

QUESTIONS = {q.id: q for q in load_questions()}


def test_load_questions_have_valid_workflows():
    # Arrange / Act / Assert
    assert len(QUESTIONS) >= 20
    for q in QUESTIONS.values():
        assert q.workflow is None or q.workflow in WORKFLOWS


def test_generic_policy_answers_every_question_at_small_scale(lab):
    # Arrange / Act
    results = [lab.run(q, "generic") for q in QUESTIONS.values()]
    # Assert
    assert all(r.correct for r in results), [r.qid for r in results if not r.correct]


def test_fine_grained_policy_answers_multi_hop_join(lab):
    # Arrange
    q = QUESTIONS["J3"]
    # Act
    r = lab.run(q, "fine_grained")
    # Assert
    assert r.correct and r.calls == 3


def test_fine_grained_policy_overflows_small_budget(lab):
    # Arrange
    q = QUESTIONS["A1"]
    # Act
    r = lab.run(q, "fine_grained", budget=20_000)
    # Assert
    assert r.status == "overflow"


def test_workflow_policy_fails_on_uncovered_question(lab):
    # Arrange
    q = QUESTIONS["L3"]
    # Act
    r = lab.run(q, "workflow")
    # Assert
    assert q.workflow is None and r.status == "no_workflow" and r.calls == 0


def test_handles_policy_answers_large_list_by_reference(lab):
    # Arrange
    q = QUESTIONS["G4"]
    # Act
    r = lab.run(q, "handles")
    # Assert
    assert r.correct and r.by_reference


def test_compaction_turns_large_result_into_wrong_answer(lab):
    # Arrange
    q = QUESTIONS["G4"]
    # Act
    plain = lab.run(q, "generic")
    compacted = lab.run(q, "generic+compact")
    # Assert
    assert plain.correct and compacted.status == "wrong"


def test_cumulative_tokens_grow_with_calls(lab):
    # Arrange
    q = QUESTIONS["J3"]
    # Act
    r = lab.run(q, "fine_grained")
    # Assert
    assert r.cumulative_input_tokens > r.turns * r.base_tokens


def test_plan_rows_counts_semi_join_steps(con):
    # Arrange
    q = QUESTIONS["J2"]
    # Act
    rows = plan_rows(con, q)
    # Assert
    assert rows["0:partsupp"] == 4 and rows["1:supplier"] == 4


def test_same_answer_tolerates_float_rounding_and_order():
    # Arrange
    gold = [(1, 10.0), (2, 20.004)]
    got = [(2, 20.0041), (1, 10.0)]
    # Act / Assert
    assert same_answer(got, gold, ordered=False)
    assert not same_answer(got, gold, ordered=True)


def test_lab_run_workflow_sql_falls_back_when_no_workflow(lab):
    # Arrange
    q = next(q for q in load_questions() if not q.workflow)
    # Act
    res = lab.run(q, "workflow+sql")
    # Assert
    assert res.status == "correct" and lab.run(q, "workflow").status == "no_workflow"
