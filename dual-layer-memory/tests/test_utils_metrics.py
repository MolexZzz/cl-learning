from __future__ import annotations

from dlmem.data.schema import Fact, label_from_counts
from dlmem.memory.qa import build_qa_prompt, extract_question, memory_lines
from dlmem.utils import exact_match, is_refusal, normalize_answer, token_f1


def test_normalize_and_em_containment():
    assert normalize_answer("The  City of Paris.") == "city of paris"
    assert exact_match("The answer is Paris.", "Paris") == 1.0
    assert exact_match("Paris, France", "Paris") == 1.0
    assert exact_match("London", "Paris") == 0.0
    assert exact_match("", "Paris") == 0.0


def test_token_f1():
    assert token_f1("the city of Paris", "city of Paris") > 0.9
    assert token_f1("London", "Paris") == 0.0


def test_refusal_detector():
    assert is_refusal("I don't know.")
    assert is_refusal("I am not sure about that.")
    assert is_refusal("")
    assert not is_refusal("The capital is Paris.")


def test_probe_prompt_roundtrip():
    prompt = build_qa_prompt("Where was X born?", ["X was born in Y."])
    assert extract_question(prompt) == "Where was X born?"
    assert memory_lines(prompt) == ["X was born in Y."]
    zero = build_qa_prompt("Where was X born?")
    assert memory_lines(zero) == []


def test_fact_label_taxonomy():
    label, rho, gamma = label_from_counts({"internal": 30}, 30)
    assert (label, rho, gamma) == ("non-write", 0.0, 0.0)

    label, rho, gamma = label_from_counts({"internal": 1, "missing": 20, "stale": 9}, 30)
    assert label == "write-new" and abs(rho - 29 / 30) < 1e-6 and gamma < 0.5

    label, rho, gamma = label_from_counts({"internal": 1, "missing": 9, "stale": 20}, 30)
    assert label == "write-update" and gamma > 0.5

    label, rho, gamma = label_from_counts({"internal": 1, "missing": 29}, 30)
    assert label == "write-new" and gamma == 0.0

    # gamma exactly 0.5 still counts as write-update (>= 0.5)
    label, _, _ = label_from_counts({"missing": 5, "stale": 5}, 10)
    assert label == "write-update"


def test_fact_em_endpoints():
    fact = Fact(
        fact_id="f",
        subject="s",
        relation="r",
        obj="o",
        text="s r o.",
        probes=[],
        counts={"internal": 3, "missing": 1, "stale": 1, "unsolved": 0},
    )
    # no probes => zero EM (defensive)
    assert fact.em_if_stored() == 0.0
    from dlmem.data.schema import Probe

    fact.probes = [Probe(f"q{i}", "o") for i in range(5)]
    assert abs(fact.em_if_stored() - 1.0) < 1e-9
    assert abs(fact.em_if_discarded() - 0.6) < 1e-9
    assert fact.write_target == 0
