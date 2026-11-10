"""``DMChainCanInterface.close`` stops its control loop before closing the bus under it.

Closing the socket first left the loop mid-send on a dead file descriptor, so an abrupt exit ended
in a spurious "motor failed" error and a traceback from the control thread.
"""

import threading
import time
from typing import Any, List, cast

from i2rt.motor_drivers.dm_driver import DMChainCanInterface


def _chain(events: List[str]) -> Any:
    chain = cast(Any, DMChainCanInterface.__new__(DMChainCanInterface))  # no socket
    chain.running = True
    chain.channel = "test"  # __repr__ reads it, as the log messages do

    def loop() -> None:
        while chain.running:
            time.sleep(0.001)
        time.sleep(0.05)  # finishing the frame in flight
        events.append("loop exited")

    chain._control_thread = threading.Thread(target=loop)
    chain._control_thread.start()
    chain.motor_interface = type("Bus", (), {"close": lambda self: events.append("bus closed")})()
    return chain


def test_close_waits_for_the_loop_before_closing_the_bus() -> None:
    events: List[str] = []
    _chain(events).close()
    assert events == ["loop exited", "bus closed"]


def test_close_without_a_started_loop_still_closes_the_bus() -> None:
    events: List[str] = []
    chain = cast(Any, DMChainCanInterface.__new__(DMChainCanInterface))
    chain.running = True
    chain.channel = "test"  # __repr__ reads it, as the log messages do
    chain.motor_interface = type("Bus", (), {"close": lambda self: events.append("bus closed")})()
    chain.close()
    assert events == ["bus closed"]


def _bus(events: List[str], fail_off: bool = False) -> Any:
    class Bus:
        def motor_off(self, motor_id: int) -> None:
            if fail_off:
                raise RuntimeError("no response")
            events.append(f"motor {motor_id} off")

        def close(self) -> None:
            events.append("bus closed")

    return Bus()


def test_listed_motors_are_disabled_after_the_loop_stops_and_before_the_bus_closes() -> None:
    events: List[str] = []
    chain = _chain(events)
    chain.motor_interface = _bus(events)
    chain.close(disable_motor_ids=[7])
    assert events == ["loop exited", "motor 7 off", "bus closed"]


def test_a_failed_disable_still_closes_the_bus() -> None:
    events: List[str] = []
    chain = cast(Any, DMChainCanInterface.__new__(DMChainCanInterface))
    chain.running = True
    chain.channel = "test"  # __repr__ reads it, as the log messages do
    chain.motor_interface = _bus(events, fail_off=True)
    chain.close(disable_motor_ids=[7])
    assert events == ["bus closed"]
