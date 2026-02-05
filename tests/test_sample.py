from pandablocks_ioc_zmq_stream import ZMQPublisher


def test_one():
    zb = ZMQPublisher(["test.hdf"], {})
    assert zb is not None
