import math

from simulations.environment import Environment
from simulations.room_model import step_room
from utils.rng import RNG


def test_environment_temperature_and_door_event():
    environment = Environment(
        base=10.0,
        amplitude=2.0,
        period_s=4.0,
        door_drop_C=3.0,
        door_start_s=1.0,
        door_duration_s=1.0,
    )

    assert math.isclose(environment.T_out(0.0), 10.0)
    assert math.isclose(environment.T_out(1.0), 9.0)
    assert math.isclose(environment.T_out(2.0), 10.0)
    assert math.isclose(environment.T_out(3.0), 8.0)


def test_room_model_euler_step():
    updated = step_room(
        T=20.0,
        heater_on=1,
        T_out=10.0,
        R=2.0,
        C=10.0,
        P=10.0,
        dt=1.0,
    )

    assert math.isclose(updated, 20.5)


def test_rng_methods_are_seeded_and_return_expected_types():
    first = RNG(123)
    second = RNG(123)

    assert first.gauss(0.0, 1.0) == second.gauss(0.0, 1.0)
    assert first.uniform(-2.0, 3.0) == second.uniform(-2.0, 3.0)
    assert first.bernoulli(0.5) == second.bernoulli(0.5)
    assert first.bernoulli(0.0) is False
    assert first.bernoulli(1.0) is True