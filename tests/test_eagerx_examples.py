from eagerx_examples.example_classifier import example_classifier
from eagerx_examples.example_full_state import example_full_state
from eagerx_examples.example_pid_only import example_pid_only


def test_example_classifier(name="test_classifier", eps=5, eval_eps=2):
    example_classifier(name=name, eps=eps, eval_eps=eval_eps)


def test_example_full_state(name="test_full_state", eps=5, eval_eps=2):
    example_full_state(name=name, eps=eps, eval_eps=eval_eps)


def test_example_pid(name="test_pid_only", eps=5, eval_eps=2):
    example_pid_only(name=name, eps=eps, eval_eps=eval_eps)
