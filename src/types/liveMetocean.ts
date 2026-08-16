export interface LiveAtmosphereCurrent {
  time: string;
  temperature_2m?: number | null;
  relative_humidity_2m?: number | null;
  apparent_temperature?: number | null;
  precipitation?: number | null;
  weather_code?: number | null;
  cloud_cover?: number | null;
  surface_pressure?: number | null;
  wind_speed_10m?: number | null;
  wind_direction_10m?: number | null;
  wind_gusts_10m?: number | null;
}

export interface LiveAtmosphereHourly {
  time: string[];
  temperature_2m: Array<number | null>;
  apparent_temperature: Array<number | null>;
  relative_humidity_2m: Array<number | null>;
  precipitation_probability: Array<number | null>;
  weather_code: Array<number | null>;
  cloud_cover: Array<number | null>;
  surface_pressure: Array<number | null>;
  wind_speed_10m: Array<number | null>;
  wind_direction_10m: Array<number | null>;
  wind_gusts_10m: Array<number | null>;
}

export interface LiveMarineCurrent {
  time: string;
  wave_height?: number | null;
  wave_direction?: number | null;
  wave_period?: number | null;
  wind_wave_height?: number | null;
  wind_wave_direction?: number | null;
  wind_wave_period?: number | null;
  swell_wave_height?: number | null;
  swell_wave_direction?: number | null;
  swell_wave_period?: number | null;
  sea_surface_temperature?: number | null;
  ocean_current_velocity?: number | null;
  ocean_current_direction?: number | null;
  sea_level_height_msl?: number | null;
}

export interface LiveMarineHourly {
  time: string[];
  wave_height: Array<number | null>;
  wave_direction: Array<number | null>;
  wave_period: Array<number | null>;
  wind_wave_height: Array<number | null>;
  wind_wave_direction: Array<number | null>;
  wind_wave_period: Array<number | null>;
  swell_wave_height: Array<number | null>;
  swell_wave_direction: Array<number | null>;
  swell_wave_period: Array<number | null>;
  sea_surface_temperature: Array<number | null>;
  ocean_current_velocity: Array<number | null>;
  ocean_current_direction: Array<number | null>;
  sea_level_height_msl: Array<number | null>;
}

export interface LiveWeatherResponse {
  latitude: number;
  longitude: number;
  timezone: string;
  timezone_abbreviation: string;
  current?: LiveAtmosphereCurrent;
  hourly?: LiveAtmosphereHourly;
}

export interface LiveMarineResponse {
  latitude: number;
  longitude: number;
  timezone: string;
  timezone_abbreviation: string;
  current?: LiveMarineCurrent;
  hourly?: LiveMarineHourly;
}

export interface LiveMetoceanData {
  atmosphere: LiveWeatherResponse | null;
  marine: LiveMarineResponse | null;
  unavailable: string[];
  fetchedAt: string;
}
