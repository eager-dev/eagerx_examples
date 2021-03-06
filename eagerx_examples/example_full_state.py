def example_full_state(name, eps, eval_eps, eps_length=200):
    # EAGERx imports
    from eagerx import Object, Bridge, initialize, log
    from eagerx.core.graph import Graph
    import eagerx.bridges.openai_gym as eagerx_gym
    import eagerx_examples  # noqa: F401

    # Start roscore & initialize main thread as node
    initialize("eagerx", anonymous=True, log_level=log.INFO)

    # Define object
    pendulum = Object.make("GymObject", "pendulum", env_id="Pendulum-v0", rate=20)

    # Define graph (agnostic) & connect nodes
    graph = Graph.create(objects=[pendulum])
    graph.connect(source=pendulum.sensors.observation, observation="observation", window=1)
    graph.connect(source=pendulum.sensors.reward, observation="reward", window=1)
    graph.connect(source=pendulum.sensors.done, observation="done", window=1)
    graph.connect(action="action", target=pendulum.actuators.action, window=1)

    # Define bridge
    bridge = Bridge.make("GymBridge", rate=20)

    # Initialize Environment (agnostic graph +  bridge)
    env = eagerx_gym.EagerxGym(name=name, rate=20, graph=graph, bridge=bridge)

    # Use stable-baselines
    import stable_baselines3 as sb

    model = sb.SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(eps * eps_length))

    # Evaluate trained policy
    for i in range(eval_eps):
        obs, done = env.reset(), False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
    env.shutdown()


if __name__ == "__main__":
    example_full_state(name="example", eps=200, eval_eps=20)
