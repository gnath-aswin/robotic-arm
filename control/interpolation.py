import numpy as np

def interpolate_lerp(t, waypoints):
    for i in range(len(waypoints)-1):
        t0, q0 = waypoints[i]
        t1, q1 = waypoints[i+1]

        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            return (1 - alpha) * np.array(q0) + alpha * np.array(q1)

    return np.array(waypoints[-1][1])


def interpolate_cubic(t, waypoints):
    for i in range(len(waypoints)-1):
        t0, q0 = waypoints[i]
        t1, q1 = waypoints[i+1]

        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)

            # cubic smoothstep
            s = 3*alpha**2 - 2*alpha**3

            return (1 - s) * np.array(q0) + s * np.array(q1)

    return np.array(waypoints[-1][1])

def circle_trajectory(t):
    center = np.array([0.3, 0.0, 0.3])
    radius = 0.2

    x = center[0]
    y = center[1] + radius * np.cos(0.1*t)
    z = center[2] + radius * np.sin(0.1*t)

    return np.array([x, y, z])