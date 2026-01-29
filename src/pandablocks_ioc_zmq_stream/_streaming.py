import logging
import os
from collections.abc import Iterator

import h5py
import numpy as np
import zmq
from pandablocks.hdf import Pipeline
from pandablocks.responses import EndData, FieldCapture, StartData


class _ZMQPublisher:
    """Class to handle publishing messages to a 0MQ PUB socket"""

    def __init__(self):
        self.socket_is_active = False
        self.socket = None

    def socket_open(self):
        if not self.socket_is_active:
            # Address is in the form tcp://<address>:<port>
            address = os.getenv("PANDABLOCKS_IOC_PUB_ZMQ_ADDRESS", None)
            if address:
                try:
                    context = zmq.Context()
                    self.socket = context.socket(zmq.PUB)
                    self.socket.bind(address)
                    self.socket_is_active = True
                    logging.info(f"0MQ PUB socket {address!r} was successfully opened.")
                except Exception as ex:
                    logging.exception(f"Failed to open 0MQ PUB socket {address!r}: {ex}")
            else:
                logging.error("The address for 0MQ PUB socket is not specified. Streaming is disabled.")

    def socket_close(self):
        if self.socket_is_active:
            self.socket.close()
            self.socket_is_active = False
            self.socket = None
            logging.info("0MQ PUB socket was closed.")

    def publish(self, message: dict):
        if self.socket_is_active:
            self.socket.send_json(message)


zmq_publisher = _ZMQPublisher()


class ZMQPublisher(Pipeline):
    def __init__(
        self,
        file_names: Iterator[str],
        capture_record_hdf_names: dict[str, dict[str, str]],
    ):
        super().__init__()
        logging.info("Initializing 0MQ Publisher ...")
        self.datasets: list[h5py.Dataset] = []
        self.capture_record_hdf_names = capture_record_hdf_names
        self.what_to_do = {
            StartData: self.stream_start,
            list: self.stream_frame,
            EndData: self.stream_stop,
        }
        self.n_emitted_frames = 0
        self.n_emitted_samples = 0

    @staticmethod
    def env_initialize():
        # The 'environment' includes the persistent instance of _ZMQPublisher.
        # The ZMQPublisher object may be created for each streamed dataset,
        # but the _ZMQPublisher should be created only once per process and
        # keep the socket open during the whole process lifetime.
        logging.info("Initializing the environment for 0MQ Publisher")
        zmq_publisher.socket_open()

    @staticmethod
    def env_cleanup():
        logging.info("Clean up of the 0MQ Publisher environment")
        zmq_publisher.socket_close()

    def get_dataset_info(self, field: FieldCapture, raw: bool):
        """
        Extract dataset info.
        """
        dataset_name = self.capture_record_hdf_names.get(field.name, {}).get(
            field.capture, f"{field.name}.{field.capture}"
        )
        dtype = field.raw_mode_dataset_dtype if raw else field.type
        dataset_info = {"name": dataset_name, "dtype": dtype}
        return dataset_info

    def stream_start(self, data: StartData):
        """
        Send 'start' message.
        Note: the function MUST pass through the 'data' without changes.
        """
        raw = data.process == "Raw"
        self.datasets = [self.get_dataset_info(field, raw) for field in data.fields]
        data_emit = {
            "msg_type": "start",
            "arm_time": data.arm_time,
            "start_time": data.start_time,
            "hw_time_offset_ns": data.hw_time_offset_ns,
        }
        self.n_emitted_frames = 0
        self.n_emitted_samples = 0

        zmq_publisher.publish(data_emit)
        logging.info("Dataset streaming: STARTED.")

        return data

    def stream_frame(self, data: list[np.ndarray]):
        """
        Send 'data' message for a single frame.
        Note: the function MUST pass through the 'data' without changes.
        """
        logging.debug("Streaming a dataframe")

        data_emit = {"msg_type": "data", "frame_number": self.n_emitted_frames, "datasets": {}}
        self.n_emitted_frames += 1

        ns = -1
        for dataset, column in zip(self.datasets, data):
            t, c = str(dataset["dtype"]), list(column)
            if ns < 0:
                ns = len(c)
            if len(c) and isinstance(c[0], np.int32):
                c = [np.float64(_) for _ in c]
                t = "float64"
            ds_emit = {"dtype": t, "size": len(c), "starting_sample_number": self.n_emitted_samples}
            ds_emit["data"] = c
            data_emit["datasets"][str(dataset["name"])] = ds_emit

        zmq_publisher.publish(data_emit)
        self.n_emitted_samples += ns

        return data

    def stream_stop(self, data: EndData):
        """
        Send 'stop' message.
        Note: the function MUST pass through the 'data' without changes.
        """
        data_emit = {"msg_type": "stop", "emitted_frames": self.n_emitted_frames}
        zmq_publisher.publish(data_emit)
        logging.info("Dataset streaming: STOPPED.")
        return data
