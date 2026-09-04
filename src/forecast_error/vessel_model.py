import numpy as np

class VesselModel:
    """
    Модель парусной яхты.
    Полярная диаграмма – зависимость скорости от угла к истинному ветру.
    """
    def __init__(self, polar_table=None):
        if polar_table is None:
            self.angles = np.array([0, 30, 45, 60, 75, 90, 120, 150, 180])
            self.speeds = np.array([0, 3.5, 5.0, 6.0, 6.5, 6.0, 4.5, 3.0, 0])
        else:
            self.angles = np.array(polar_table['angle'])
            self.speeds = np.array(polar_table['speed'])
        self.speed_func = lambda angle: np.interp(angle, self.angles, self.speeds, left=0, right=0)
    
    def get_speed(self, true_wind_speed, true_wind_direction, heading,
                  current_speed=0, current_direction=0, wave_height=0):
        angle_deg = (true_wind_direction - heading) % 360
        if angle_deg > 180:
            angle_deg = 360 - angle_deg
        base_speed_knots = self.speed_func(angle_deg)
        speed_factor = (true_wind_speed / 10.0) ** 1.5
        vessel_speed_knots = base_speed_knots * speed_factor
        vessel_speed_ms = vessel_speed_knots * 0.5144
        current_angle = (current_direction - heading) % 360
        if current_angle > 180:
            current_angle = 360 - current_angle
        if current_angle < 90:
            vessel_speed_ms += current_speed * np.cos(np.radians(current_angle)) * 0.7
        else:
            vessel_speed_ms -= current_speed * 0.3
        if wave_height > 1.5:
            vessel_speed_ms *= (1 - 0.1 * (wave_height - 1.5))
        return max(vessel_speed_ms, 0.1)