import random
import pytest
from repair import (
    repair,
    NR_PLACES,
    NR_TRANSITIONS,
    FIRST_VISIBLE_ROW
)
 
 
# fixture 
@pytest.fixture(autouse=True)
def _seed_random():
    """Seed Python's RNG before each test so any randomness
    is reproducible. Autouse so don't have to call it every time
    """
    random.seed(42)


def test_invisible_self_loop_is_fixed():
    """ Repair the self loop in the invisible row ."""
    ind = [3, 3] + [0, 1] * 11
    repair(ind)
    assert ind[0] != ind[1]
 
 
def test_invisible_backward_is_left_alone():
    """ Don't penalise the backward step in the invisible row.
    """
    ind = [7, 2] + [0, 1] * 11
    repair(ind)
    assert ind[:2] == [7, 2]
 
 
def test_visible_self_loop_is_fixed():
    """ Repair the self loop in the visible row ."""
    ind = [0, 1, 0, 1, 5, 5] + [0, 1] * 9
    repair(ind)
    assert ind[4] < ind[5]
 
 
def test_visible_backward_is_fixed():
    """ Repair the backward step in the visible row."""
    ind = [0, 1, 0, 1, 7, 3] + [0, 1] * 9
    repair(ind)
    assert ind[4] < ind[5]
 
 
def test_repair_returns_same_object():
    """ Make sure the repair mutates in place and returns the same 
    list in memory and not a copy ."""
    ind = [3, 3] + [0, 1] * 11
    out = repair(ind)
    assert out is ind

def is_valid(ind):
    """ Verify both constraints for the individual  ."""
    for m in range(NR_TRANSITIONS):
        a_in, a_out = ind[2 * m], ind[2 * m + 1]
        if a_in == a_out:
            return False
        if m >= FIRST_VISIBLE_ROW and a_in >= a_out:
            return False
    return True
 
 
# Stress test over 1000 chromosomes
@pytest.mark.parametrize("trial", range(1000))
def test_random_chromosome_is_repaired(trial):
    """
    1000 fully random chromosomes must all end up valid after repair.
    The re-seeding per trial is deliberate to ensure that we can debug 
    if test fails somewhere specific 
    """
    random.seed(trial)
    ind = [random.randint(0, NR_PLACES - 1)
           for _ in range(NR_TRANSITIONS * 2)]
    repair(ind)
    assert is_valid(ind)
 

 
 
