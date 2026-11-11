def controller(v, target_speed):
    # Accelerate when too slow; brake when too fast.
    acceleration = 1.0 * (target_speed - v)
    return max(-5.0, min(acceleration, 2.0))


def safety_filter(requested_a, x, v, obstacle_x):
    braking = 5.0
    buffer = 5.0

    # Predict one step using the requested action.
    next_x = x + v * dt + 0.5 * requested_a * dt**2
    next_v = max(0.0, v + requested_a * dt)

    next_gap = obstacle_x - next_x
    stopping_distance = next_v**2 / (2 * braking)

    if next_gap <= stopping_distance + buffer:
        return -braking

    return requested_a


x = 0.0
v = 20.0

target_speed = 20.0
obstacle_x = 60.0

dt = 0.02
duration = 8.0
first_override = True

for step in range(round(duration / dt)):
    requested_a = controller(v, target_speed)
    applied_a = safety_filter(requested_a, x, v, obstacle_x)

    if applied_a < requested_a and first_override:
        print(f"Filter starts braking at x = {x:.2f} m")
        first_override = False

    moving_time = dt
    if applied_a < 0.0:
        moving_time = min(dt, v / abs(applied_a))

    x += v * moving_time + 0.5 * applied_a * moving_time**2
    v = max(0.0, v + applied_a * moving_time)

    if x >= obstacle_x:
        print("Collision!")
        break

    if v == 0.0:
        print(f"Stopped at x = {x:.2f} m")
        print(f"Gap to obstacle: {obstacle_x - x:.2f} m")
        break