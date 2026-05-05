import random
from support import NR_PLACES, NR_TRANSITIONS

FIRST_VISIBLE_ROW = 2    # rows 0 and 1 are the invisble 

def repair(individual):
    """Repair the individual in-place so that C3, for every row and C5,
    for visible rows only are satisfied.  Returns the same individual
    for convenience.
    """
    for m in range(NR_TRANSITIONS):
        i_in, i_out = 2 * m, 2 * m + 1
        a_in, a_out = individual[i_in], individual[i_out]
        is_visible  = m >= FIRST_VISIBLE_ROW

        if is_visible:
            # for visible rows enforce strict ordering a_in < a_out
            # resample uniformly over the valid (a_in, a_out) pairs
            # whenever the row is not already vali
            if a_in >= a_out:
                new_in  = random.randint(0, NR_PLACES - 2)
                new_out = random.randint(new_in + 1, NR_PLACES - 1)
                individual[i_in]  = new_in
                individual[i_out] = new_out
        else:
            # for invisible row enforce only C3 - no self loop 
            if a_in == a_out:
                # create a range of suitable options
                choices = [v for v in range(NR_PLACES) if v != a_in]
                # assing one of these choices to i_out
                individual[i_out] = random.choice(choices)

    return individual

def constraint_repair():
    """Return a DEAP decorator that repairs every offspring.
        strucutured exactly like https://deap.readthedocs.io/en/master/tutorials/basic/part2.html
    #tool-decoration)"""
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                repair(child)
            return offspring
        return wrapper
    return decorator





