import math
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    fr = 48000
    dur = 1.0

    _10db_dim_time = 0.2  # Time it takes for the hair to diminish by 10db, in seconds
    FRICTION_COF = pow(0.1, _10db_dim_time * fr)

    for sine_hz in range(50, 100, 2):
        MYSTERY = pow(math.tau * sine_hz / fr, 2)

        hair_pos = 0.0
        hair_speed = 0.0

        total_speed = 0.0
        total_pos = 0.0

        for sine_speed in [math.sin((sine_hz * math.tau * x) / fr) for x in range(math.floor(dur * fr))]:
            acc_center_pull = - MYSTERY * pow(hair_pos, 1)
            acc_friction = 0 * hair_speed
            acc_signal = sine_speed

            hair_speed += acc_center_pull + acc_friction + acc_signal
            hair_pos += hair_speed

            total_speed += abs(hair_speed) / fr
            total_pos += abs(hair_pos) / fr

        print(total_speed)