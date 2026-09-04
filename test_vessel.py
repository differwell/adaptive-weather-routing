import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.forecast_error.vessel_model import VesselModel

if __name__ == "__main__":
    vessel = VesselModel()
    speed = vessel.get_speed(
        true_wind_speed=12,
        true_wind_direction=270,
        heading=0,
        current_speed=0.5,
        current_direction=180,
        wave_height=2.0
    )
    print(f"Скорость яхты: {speed:.2f} м/с")