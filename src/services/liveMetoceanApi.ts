import type { LiveMarineResponse, LiveMetoceanData, LiveWeatherResponse } from 'types/liveMetocean';

const TANGER_MED = { latitude: 35.891, longitude: -5.501 };

const WEATHER_VARIABLES = [
  'temperature_2m',
  'relative_humidity_2m',
  'apparent_temperature',
  'precipitation',
  'weather_code',
  'cloud_cover',
  'surface_pressure',
  'wind_speed_10m',
  'wind_direction_10m',
  'wind_gusts_10m',
];

const WEATHER_HOURLY = [
  'temperature_2m',
  'apparent_temperature',
  'relative_humidity_2m',
  'precipitation_probability',
  'weather_code',
  'cloud_cover',
  'surface_pressure',
  'wind_speed_10m',
  'wind_direction_10m',
  'wind_gusts_10m',
];

const MARINE_VARIABLES = [
  'wave_height',
  'wave_direction',
  'wave_period',
  'wind_wave_height',
  'wind_wave_direction',
  'wind_wave_period',
  'swell_wave_height',
  'swell_wave_direction',
  'swell_wave_period',
  'sea_surface_temperature',
  'ocean_current_velocity',
  'ocean_current_direction',
  'sea_level_height_msl',
];

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { headers: { Accept: 'application/json' }, signal });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Open-Meteo request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

const weatherUrl = () => {
  const params = new URLSearchParams({
    latitude: String(TANGER_MED.latitude),
    longitude: String(TANGER_MED.longitude),
    current: WEATHER_VARIABLES.join(','),
    hourly: WEATHER_HOURLY.join(','),
    past_hours: '24',
    forecast_hours: '72',
    timezone: 'Africa/Casablanca',
    wind_speed_unit: 'ms',
    cell_selection: 'nearest',
  });
  return `https://api.open-meteo.com/v1/forecast?${params.toString()}`;
};

const marineUrl = () => {
  const params = new URLSearchParams({
    latitude: String(TANGER_MED.latitude),
    longitude: String(TANGER_MED.longitude),
    current: MARINE_VARIABLES.join(','),
    hourly: MARINE_VARIABLES.join(','),
    past_hours: '24',
    forecast_hours: '72',
    timezone: 'Africa/Casablanca',
    cell_selection: 'sea',
  });
  return `https://marine-api.open-meteo.com/v1/marine?${params.toString()}`;
};

export const liveMetoceanApi = {
  async getDashboard(signal?: AbortSignal): Promise<LiveMetoceanData> {
    const [atmosphereResult, marineResult] = await Promise.allSettled([
      getJson<LiveWeatherResponse>(weatherUrl(), signal),
      getJson<LiveMarineResponse>(marineUrl(), signal),
    ] as const);

    const unavailable: string[] = [];
    if (atmosphereResult.status === 'rejected') unavailable.push('atmosphere Open-Meteo');
    if (marineResult.status === 'rejected') unavailable.push('marine Open-Meteo');

    return {
      atmosphere: atmosphereResult.status === 'fulfilled' ? atmosphereResult.value : null,
      marine: marineResult.status === 'fulfilled' ? marineResult.value : null,
      unavailable,
      fetchedAt: new Date().toISOString(),
    };
  },
};
