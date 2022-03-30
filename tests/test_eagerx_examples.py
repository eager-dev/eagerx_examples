def test_example_full_state(name="test_full_state", eps=1, eval_eps=1, eps_length=1):
    from eagerx_examples.example_full_state import example_full_state

    example_full_state(name=name, eps=eps, eval_eps=eval_eps, eps_length=eps_length)


def test_example_pid(name="test_pid_only", eps=1, eval_eps=1, eps_length=1):
    from eagerx_examples.example_pid_only import example_pid_only

    example_pid_only(name=name, eps=eps, eval_eps=eval_eps, eps_length=eps_length)


def test_example_classifier(name="test_classifier", eps=1, eval_eps=1, eps_length=1, classifier_epochs=1):
    from pyvirtualdisplay import Display

    display = Display(visible=False, size=(1400, 900))
    display.start()
    from eagerx_examples.example_classifier import example_classifier

    example_classifier(name=name, eps=eps, eval_eps=eval_eps, eps_length=eps_length, classifier_epochs=classifier_epochs)
    display.stop()
