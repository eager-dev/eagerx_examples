def test_example_full_state(name="test_full_state", eps=1, eval_eps=1):
    from eagerx_examples.example_full_state import example_full_state
    example_full_state(name=name, eps=eps, eval_eps=eval_eps)


def test_example_pid(name="test_pid_only", eps=1, eval_eps=1):
    from eagerx_examples.example_pid_only import example_pid_only
    example_pid_only(name=name, eps=eps, eval_eps=eval_eps)