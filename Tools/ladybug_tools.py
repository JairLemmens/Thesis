import ladybug
import ladybug_geometry
import ladybug_radiance
import ladybug.epw as epw
from ladybug_radiance.skymatrix import SkyMatrix
from ladybug_radiance.visualize.skydome import SkyDome
import numpy as np
import ladybug.analysisperiod as ap


class Climate():
    def __init__(self, epw_file):
        epw_data = epw.EPW(epw_file)
        self.location = epw_data.location
        self.dry_bulb_temperature = epw_data.dry_bulb_temperature
        self.dew_point_temperature = epw_data.dew_point_temperature
        self.relative_humidity = epw_data.relative_humidity
        self.wind_speed = epw_data.wind_speed
        self.wind_direction = epw_data.wind_direction
        self.direct_normal_rad = epw_data.direct_normal_radiation
        self.diffuse_horizontal_rad = epw_data.diffuse_horizontal_radiation
        self.global_horizontal_rad = epw_data.global_horizontal_radiation
        self.horizontal_infrared_rad = epw_data.horizontal_infrared_radiation_intensity
        self.direct_normal_ill = epw_data.direct_normal_illuminance
        self.diffuse_horizontal_ill = epw_data.diffuse_horizontal_illuminance
        self.global_horizontal_ill = epw_data.global_horizontal_illuminance
        self.total_sky_cover = epw_data.total_sky_cover
        self.barometric_pressure = epw_data.atmospheric_station_pressure
        self.model_year = epw_data.years
        self.g_temp = epw_data.monthly_ground_temperature
        self.ground_temperature = [self.g_temp[key] for key in sorted(self.g_temp.keys())]
        self.sky_mtx = SkyMatrix.from_components(self.location, self.direct_normal_rad, self.diffuse_horizontal_rad)
        self.sky_dome = SkyDome(self.sky_mtx)
        self.sky_vectors = np.array(self.sky_dome.patch_vectors)
        self.intensities = np.array(self.sky_dome.total_values)


    def intensities_from_hoys(self,start_month= 1,start_day= 1,start_hour=0,end_month=12,end_day=31,end_hour=23,timestep=1):
        _hoys=ap.AnalysisPeriod(start_month, start_day, start_hour,end_month, end_day, end_hour, timestep).hoys
        _sky_dome = SkyDome(SkyMatrix.from_components(self.location, self.direct_normal_rad, self.diffuse_horizontal_rad,hoys=_hoys))
        _intensities = np.array(_sky_dome.total_values)
        return(_intensities)