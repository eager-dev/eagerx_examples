from typing import Optional, List
from collections import deque
import numpy as np

# IMPORT ROS
from std_msgs.msg import Float32MultiArray

# IMPORT EAGERX
import eagerx.core.register as register
from eagerx.utils.utils import Msg
from eagerx.core.entities import Node, Processor, SpaceConverter
from eagerx.core.constants import process


class PidController(Node):
    @staticmethod
    @register.spec("PidController", Node)
    def spec(
        spec,
        name: str,
        rate: float,
        gains: List[float],
        y_range: Optional[List[float]] = None,
        u_range: Optional[List[float]] = None,
        index: int = 0,
        u0: float = 0.0,
        process: Optional[int] = process.NEW_PROCESS,
        color: Optional[str] = "grey",
    ):
        """
        PID controller.

        :param spec: Not provided by user.
        :param name: Node name
        :param rate: Rate at which callback is called.
        :param y_range: reference range [min, max]. This is not enforced, but merely used for space_converters.
        :param u_range: control range [min, max]. This is not enforced, but merely used for space_converters.
        :param gains: [Kp, Kd, Ki]
        :param index: Index (related to Float32MultiArray.data[index])
        :param u0: initial action
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: BRIDGE, 3: EXTERNAL}
        :param color: console color of logged messages. {'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'grey'}
        :return:
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(PidController)

        # Set default
        y_range = y_range if y_range else [-1.0, 1.0]
        u_range = u_range if u_range else [-1.0, 1.0]

        # Modify default node params
        params = dict(name=name, rate=rate, process=process, color=color, inputs=["y", "yref"], outputs=["u"])
        spec.config.update(params)

        # Modify custom node params
        spec.config.gains = gains
        spec.config.u0 = u0

        # Add converter & space_converter to outputs
        c = Processor.make("GetIndex_Float32MultiArray", index=index)
        spec.outputs.u.converter = c
        spec.outputs.u.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", [u_range[0]], [u_range[1]], dtype="float32"
        )

        # Add space_converter to inputs
        sc = SpaceConverter.make("Space_Float32MultiArray", [y_range[0]], [y_range[1]], dtype="float32")
        spec.inputs.y.space_converter = sc
        spec.inputs.yref.space_converter = sc
        spec.inputs.y.converter = c
        spec.inputs.yref.converter = c

    def initialize(self, gains, u0):
        self.controller = PID(u0=u0, kp=gains[0], kd=gains[1], ki=gains[2], dt=1 / self.rate)

    @register.states()
    def reset(self):
        self.controller.reset()

    @register.inputs(y=Float32MultiArray, yref=Float32MultiArray)
    @register.outputs(u=Float32MultiArray)
    def callback(self, t_n: float, y: Optional[Msg] = None, yref: Optional[Msg] = None):
        trig = y.msgs[-1].data[:2]
        y = np.arctan2(trig[1], trig[0])
        yref = yref.msgs[-1].data[0]
        u = self.controller.next_action(y, ref=yref)
        return dict(u=Float32MultiArray(data=np.array([u])))


class PID:
    def __init__(self, u0: float, kp: float, kd: float, ki: float, dt: float):
        self.u0 = u0
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.dt = dt

        self.F = [kp + ki * dt + kd / dt, -kp - 2 * kd / dt, kd / dt]

        self.window = None
        self.u = None

    def reset(self):
        self.u = self.u0
        self.window = deque(maxlen=3)

    def next_action(self, y: float, ref: float = 0.0):
        # Add error
        self.window.appendleft(ref - y)

        # Calculate action
        for idx, e in enumerate(self.window):
            self.u += self.F[idx] * e

        return self.u
