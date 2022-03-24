def example_pid_only(name, eps, eval_eps):
    # EAGERx imports
    from eagerx import Object, Bridge, Node, initialize, log, process
    from eagerx.core.graph import Graph
    import eagerx.bridges.openai_gym as eagerx_gym
    import eagerx_examples  # noqa: F401

    # Start roscore & initialize main thread as node
    initialize("eagerx", anonymous=True, log_level=log.INFO)

    # Define object
    pendulum = Object.make("GymObject", "pendulum", env_id="Pendulum-v0", rate=20)

    # Define PID controller & classifier
    pid = Node.make("PidController", "pid", rate=20, gains=[8, 1, 0], y_range=[-4, 4])

    # Define graph (agnostic) & connect nodes
    graph = Graph.create(nodes=[pid], objects=[pendulum])
    graph.connect(source=pendulum.sensors.reward, observation="reward")
    graph.connect(source=pendulum.sensors.done, observation="done")
    graph.connect(source=pendulum.sensors.observation, observation="state")
    # Connect PID
    graph.connect(source=pendulum.sensors.observation, target=pid.inputs.y)
    graph.connect(action="yref", target=pid.inputs.yref)
    graph.connect(source=pid.outputs.u, target=pendulum.actuators.action)

    # Define bridge
    bridge = Bridge.make("GymBridge", rate=20)

    # Initialize Environment (agnostic graph +  bridge)
    env = eagerx_gym.EagerxGym(name=name, rate=20, graph=graph, bridge=bridge)

    # Use stable-baselines
    import stable_baselines3 as sb

    model = sb.SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(eps * 200))

    # Evaluate trained policy
    for i in range(eval_eps):
        obs, done = env.reset(), False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
    env.shutdown()


if __name__ == "__main__":
    example_pid_only(name="example", eps=150, eval_eps=20)
