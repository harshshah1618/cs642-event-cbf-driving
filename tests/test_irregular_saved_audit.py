import pytest
from tools.irregular_audit_saved import collision_bounds


def episode(cost, **flags):
    return dict(cost=cost, **(dict(crash_vehicle=False, crash_object=False, crash_human=False) | flags))


def test_one_unit_cannot_distinguish_collision_from_terminal_departure():
    assert collision_bounds(episode(1)) == (0, 1)


def test_multiple_cost_units_prove_contact_even_when_final_flags_are_false():
    assert collision_bounds(episode(2)) == (1, 1)
    assert collision_bounds(episode(76)) == (1, 1)


def test_zero_cost_requires_human_flag_check():
    assert collision_bounds(episode(0)) == (0, 0)
    assert collision_bounds(episode(0, crash_human=True)) == (1, 1)


def test_final_contact_resolves_a_cost_one_episode():
    assert collision_bounds(episode(1, crash_object=True)) == (1, 1)


@pytest.mark.parametrize("cost", [-1, .5, float("nan"), float("inf")])
def test_invalid_native_cost_rejects_inference(cost):
    with pytest.raises(ValueError):
        collision_bounds(episode(cost))
