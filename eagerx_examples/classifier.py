from typing import Dict, List, Union, Optional
import matplotlib.pyplot as plt
import numpy as np
import h5py
import warnings
from collections import deque

# IMPORT ROS
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image

# IMPORT EAGERX
import eagerx.core.register as register
from eagerx.utils.utils import Msg
from eagerx.core.entities import Node, Processor, SpaceConverter
from eagerx.core.constants import process


class Classifier(Node):
    @staticmethod
    @register.spec("Classifier", Node)
    def spec(
        spec,
        name: str,
        rate: float,
        cam_rate: float,
        data: str,
        show: bool = False,
        window: int = 3,
        state_range: Optional[List[List[float]]] = None,
        process: Optional[int] = process.BRIDGE,
        color: Optional[str] = "grey",
    ):
        """
        Classifier.

        :param spec: Not provided by user.
        :param name: Node name
        :param rate: Rate at which callback is called.
        :param cam_rate: Rate at which images are produced
        :param data: Path to dataste
        :param show: Flag to plot train results
        :param window: Default window length of image input.
        :param state_range: state range [[min], [max]]. This is not enforced, but merely used for space_converters.
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: BRIDGE, 3: EXTERNAL}
        :param color: console color of logged messages. {'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'grey'}
        :return:
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(Classifier)

        # Set default
        state_range = state_range if state_range else [[-1.0, -1.0, -8.0], [1.0, 1.0, 8.0]]

        # Modify default node params
        params = dict(name=name, rate=rate, process=process, color=color, inputs=["image"], outputs=["state"])
        spec.config.update(params)

        # Modify custom node params
        spec.config.update({"cam_rate": cam_rate, "data": data, "show": show})

        # Set window length of images
        spec.inputs.image.window = window

        # Add space_converter to output
        spec.outputs.state.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", state_range[0], state_range[1], dtype="float32"
        )

    def initialize(self, cam_rate, data, show):
        self.cam_rate = cam_rate
        # Prepare data
        train, test = prepare_dataset(data, show=show)
        # Train model
        self.classifier = Model()
        self.classifier.train_model(train, test, show=show, epochs=30, batch_size=64)

    @register.states()
    def reset(self):
        pass

    @register.inputs(image=Image)
    @register.outputs(state=Float32MultiArray)
    def callback(self, t_n: float, image: Optional[Msg] = None):
        # Prepare image batch
        num_images = len(image.msgs)
        prebatch = bytearray()
        for ros_im in image.msgs:
            if isinstance(ros_im.data, bytes):
                prebatch.extend(ros_im.data)
            else:
                raise NotImplementedError("Only byte arrays are supported.")

        stacked = np.frombuffer(prebatch, dtype=np.uint8).reshape(num_images, 28, 28, -1)

        # Calculate state
        stacked_float32 = change_dtype_np_image(stacked, "float32")
        trig = self.classifier.predict(stacked_float32)
        angles = np.arctan2(
            trig[:, 1],
            trig[:, 0],
        )
        diff = np.diff(angles)
        diff_wrapped = np.arctan2(np.sin(diff), np.cos(diff))
        thdot = diff_wrapped.mean() * self.cam_rate if len(diff_wrapped) > 0 else 0.0
        state = np.array([trig[-1, 0], trig[-1, 1], thdot], dtype="float32")
        return dict(state=Float32MultiArray(data=state))


def prepare_dataset(data_path, show=False, num_examples=12000):
    with h5py.File(data_path, "r") as hf:
        observation = hf["observation.h5"][:]
        state = hf["state.h5"][:]
    print("Loaded observation data: %s" % str(observation.shape))
    print("Loaded state data: %s" % str(state.shape))

    # DATASET PARAMETERS
    observation = observation[:num_examples]
    state = state[:num_examples]

    # DATA PRE-PROCESSING
    # Scale pixel values to a range of 0 to 1 before feeding them to the neural network model.
    observation = observation.astype(np.float32) / 255.0

    # CREATE TEST DATASET
    test_split = 0.2
    if 0 < test_split < 1:
        split_at = int(len(observation) * (1 - test_split))
    else:
        raise ValueError("Must hold-out data as a test dataset. Set parameter 0 < test_split < 1.")
    test_obs = observation[split_at:, :, :, :]
    test_theta = state[split_at:, 0]
    test_trig = np.hstack([np.cos(test_theta)[:, None], np.sin(test_theta)[:, None]])

    # CREATE TRAINING DATASET
    train_obs = observation[:split_at, :, :, :]
    train_theta = state[:split_at, 0]
    train_trig = np.hstack([np.cos(train_theta)[:, None], np.sin(train_theta)[:, None]])

    # VERIFY TRAINING DATA
    # To verify that the data is in the correct format and that you're ready to build and train the network,
    # let's display the first 25 images from the dataset and display the corresponding theta value below each image.
    if show:
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_obs[i])
            plt.xlabel(str(round(train_theta[i] / np.pi, 2)) + "$\\pi$")
    return (train_obs, train_theta, train_trig), (test_obs, test_theta, test_trig)


def change_dtype_np_image(image, dtype):
    if dtype and image.dtype != dtype:
        if image.dtype in ("float32", "float64") and dtype == "uint8":
            image = (image * 255).astype(dtype)
        elif image.dtype == "uint8" and dtype in ("float32", "float64"):
            image = image.astype(dtype) / 255
        else:
            message = "Cannot convert observations from {} to {}."
            raise NotImplementedError(message.format(image.dtype, dtype))
    return image


def show_ros_image(msg):
    import matplotlib.pyplot as plt

    rgb_back = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    plt.imshow(rgb_back)
    plt.show()


# Torch related
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(5408, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x

    def predict(self, x):
        if isinstance(x, np.ndarray):
            is_numpy = True
            x = torch.from_numpy(x).permute([0, 3, 1, 2])
        else:
            is_numpy = False
        output = self.forward(x)
        if is_numpy:
            return output.detach().numpy()
        else:
            return output

    def train_model(self, train, test, shuffle=True, batch_size=64, epochs=30, show=False):
        warnings.filterwarnings("ignore")

        train_obs = torch.from_numpy(train[0]).permute([0, 3, 1, 2])
        train_theta = torch.from_numpy(train[1])
        train_trig = torch.from_numpy(train[2])
        test_obs = torch.from_numpy(test[0]).permute([0, 3, 1, 2])
        test_theta = torch.from_numpy(test[1])
        test_trig = torch.from_numpy(test[2])

        # PREPARE TORCH DATASET & DATALOADER
        trainset = TensorDataset(train_obs, train_trig)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        testset = TensorDataset(test_obs, test_trig)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        # DEFINE LOSS AND OPTIMIZER
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 150 == 149:  # print every 2000 mini-batches
                    rospy.loginfo(f"[epoch={epoch + 1}, mini-batch={i+1}] loss: {running_loss/2000: .5f}")
                    running_loss = 0.0

        print("Finished Training")

        # VALIDATE MODEL
        output = self.predict(test_obs).detach().numpy()
        pred_theta = np.arctan2(output[:, 1], output[:, 0])
        self._validate_model(pred_theta, test_theta, str_model_type="$M^{cnn}$", show=show)

        # MODEL SUMMARY
        if show:
            plt.show()

    def _validate_model(self, pred_theta, test_theta, str_model_type="model_cnn", show=False):
        # EVALUATE MODEL ACCURACY
        # Calculate average error per bin over theta range [-pi, pi]
        test_error = np.abs(test_theta - pred_theta)
        test_error[test_error > np.pi] -= 2 * np.pi
        test_error = np.abs(test_error)
        bins = np.linspace(-np.pi, np.pi, 21)
        digitized = np.digitize(test_theta, bins)
        bin_means = np.array([test_error[digitized == i].mean() for i in range(1, len(bins))])
        if show:
            fig, ax = plt.subplots()
            ax.bar(bins[:-1], bin_means, width=np.diff(bins), edgecolor="black", align="edge")
            ax.set_xlabel("$\\theta$ (rad)")
            ax.set_ylabel("$|\\bar{\\theta} -\\theta|$ (rad)")
            ax.set_title("%s - Average prediction error %s" % (str_model_type, "{:.2e}".format(test_error.mean())))
