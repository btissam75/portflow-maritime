import { Box, Button, Stack, Typography } from '@mui/material';
import { useMemo, useState } from 'react';
import * as echarts from 'echarts/core';
import { GeoComponent, TooltipComponent } from 'echarts/components';
import { EffectScatterChart, LinesChart, ScatterChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import world from 'assets/json/world.json';
import IconifyIcon from 'components/base/IconifyIcon';
import ReactEchart from 'components/base/ReactEhart';
import type {
  LiveAtmosphereCurrent,
  LiveAtmosphereHourly,
  LiveMarineCurrent,
  LiveMarineHourly,
} from 'types/liveMetocean';
import type { MetoceanVesselImpact } from 'types/metocean';
import { portflowPalette as pf } from 'theme/portflowPalette';

echarts.use([
  TooltipComponent,
  GeoComponent,
  LinesChart,
  ScatterChart,
  EffectScatterChart,
  CanvasRenderer,
]);

// @ts-ignore The bundled GeoJSON follows ECharts' map schema.
echarts.registerMap('portflow-strait', { geoJSON: world });

type MapLayer = 'WAVES' | 'WIND' | 'RAIN' | 'VESSELS';
type MapZone = 'STRAIT' | 'ATLANTIC' | 'PORT' | 'MEDITERRANEAN';

const TANGER_MED: [number, number] = [-5.501, 35.891];
const ALGECIRAS: [number, number] = [-5.448, 36.132];

const layerDefinitions: Record<
  MapLayer,
  { label: string; icon: string; color: string; soft: string }
> = {
  WAVES: {
    label: 'Vagues',
    icon: 'lucide:waves',
    color: '#35D7F2',
    soft: 'rgba(53,215,242,.17)',
  },
  WIND: {
    label: 'Vent',
    icon: 'lucide:wind',
    color: '#FFC44D',
    soft: 'rgba(255,196,77,.17)',
  },
  RAIN: {
    label: 'Pluie',
    icon: 'lucide:cloud-rain',
    color: '#8D91FF',
    soft: 'rgba(141,145,255,.18)',
  },
  VESSELS: {
    label: 'Navires',
    icon: 'lucide:ship',
    color: '#54E3B2',
    soft: 'rgba(84,227,178,.17)',
  },
};

const zoneDefinitions: Record<
  MapZone,
  { label: string; short: string; center: [number, number]; zoom: number }
> = {
  ATLANTIC: { label: 'Approche Atlantique', short: 'Atlantique', center: [-5.78, 35.91], zoom: 16 },
  STRAIT: { label: 'Détroit de Gibraltar', short: 'Détroit', center: [-5.53, 35.98], zoom: 19 },
  PORT: { label: 'Zone Tanger Med', short: 'Port', center: [-5.51, 35.9], zoom: 31 },
  MEDITERRANEAN: {
    label: 'Approche Méditerranée',
    short: 'Méditerranée',
    center: [-5.27, 35.99],
    zoom: 16,
  },
};

const format = (value: number | null | undefined, digits = 1) =>
  value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits);

const cardinal = (degrees: number | null | undefined) => {
  if (degrees == null || !Number.isFinite(degrees)) return '—';
  const labels = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO'];
  return labels[Math.round((((degrees % 360) + 360) % 360) / 45) % 8];
};

const hourIndex = (times: string[] | undefined, hour: number) => {
  if (!times?.length) return -1;
  const target = Date.now() + hour * 60 * 60 * 1000;
  return times.reduce((best, time, index) => {
    const distance = Math.abs(new Date(time).getTime() - target);
    const bestDistance = Math.abs(new Date(times[best]).getTime() - target);
    return distance < bestDistance ? index : best;
  }, 0);
};

const valueAt = (
  times: string[] | undefined,
  values: Array<number | null> | undefined,
  hour: number,
) => {
  const index = hourIndex(times, hour);
  return index < 0 ? null : values?.[index] ?? null;
};

const vesselPosition = (id: string, index: number): [number, number] => {
  const seed = [...id].reduce((total, character) => total + character.charCodeAt(0), index * 17);
  const lane = index % 2 === 0 ? 1 : -1;
  return [-5.9 + (seed % 32) * 0.008, 35.84 + lane * ((seed % 11) * 0.004)];
};

const weatherPresentation = (code: number | null | undefined) => {
  if (code == null)
    return {
      label: 'Conditions à confirmer',
      icon: 'lucide:cloud',
      color: '#A9C7D2',
      glow: 'rgba(169,199,210,.12)',
    };
  if (code === 0)
    return {
      label: 'Ciel dégagé',
      icon: 'lucide:sun',
      color: '#FFD35A',
      glow: 'rgba(255,211,90,.25)',
    };
  if (code <= 3)
    return {
      label: 'Éclaircies et nuages',
      icon: 'lucide:cloud-sun',
      color: '#F7C96A',
      glow: 'rgba(247,201,106,.18)',
    };
  if (code <= 48)
    return {
      label: 'Brume côtière',
      icon: 'lucide:cloud-fog',
      color: '#B5C9D3',
      glow: 'rgba(181,201,211,.18)',
    };
  if (code <= 82)
    return {
      label: 'Pluie possible',
      icon: 'lucide:cloud-rain',
      color: '#8D91FF',
      glow: 'rgba(141,145,255,.22)',
    };
  return {
    label: 'Risque orageux',
    icon: 'lucide:cloud-lightning',
    color: '#FF8A65',
    glow: 'rgba(255,138,101,.24)',
  };
};

interface MetoceanSituationMapProps {
  atmosphere?: LiveAtmosphereCurrent;
  atmosphereHourly?: LiveAtmosphereHourly;
  marine?: LiveMarineCurrent;
  marineHourly?: LiveMarineHourly;
  impacts?: MetoceanVesselImpact[];
  forecastHour?: number;
  height?: number;
}

const MetoceanSituationMap = ({
  atmosphere,
  atmosphereHourly,
  marine,
  marineHourly,
  impacts = [],
  forecastHour = 0,
  height = 560,
}: MetoceanSituationMapProps) => {
  const [layer, setLayer] = useState<MapLayer>('WAVES');
  const [zone, setZone] = useState<MapZone>('STRAIT');
  const activeLayer = layerDefinitions[layer];
  const activeZone = zoneDefinitions[zone];
  const isForecast = forecastHour > 0;

  const forecastTemperature =
    forecastHour === 0
      ? atmosphere?.temperature_2m
      : valueAt(atmosphereHourly?.time, atmosphereHourly?.temperature_2m, forecastHour);
  const forecastWeatherCode =
    forecastHour === 0
      ? atmosphere?.weather_code
      : valueAt(atmosphereHourly?.time, atmosphereHourly?.weather_code, forecastHour);
  const forecastWind =
    forecastHour === 0
      ? atmosphere?.wind_speed_10m
      : valueAt(atmosphereHourly?.time, atmosphereHourly?.wind_speed_10m, forecastHour);
  const forecastWindDirection =
    forecastHour === 0
      ? atmosphere?.wind_direction_10m
      : valueAt(atmosphereHourly?.time, atmosphereHourly?.wind_direction_10m, forecastHour);
  const forecastGust =
    forecastHour === 0
      ? atmosphere?.wind_gusts_10m
      : valueAt(atmosphereHourly?.time, atmosphereHourly?.wind_gusts_10m, forecastHour);
  const forecastRain = valueAt(
    atmosphereHourly?.time,
    atmosphereHourly?.precipitation_probability,
    forecastHour,
  );
  const forecastWave =
    forecastHour === 0
      ? marine?.wave_height
      : valueAt(marineHourly?.time, marineHourly?.wave_height, forecastHour);
  const forecastSwell =
    forecastHour === 0
      ? marine?.swell_wave_height
      : valueAt(marineHourly?.time, marineHourly?.swell_wave_height, forecastHour);
  const forecastPeriod =
    forecastHour === 0
      ? marine?.wave_period
      : valueAt(marineHourly?.time, marineHourly?.wave_period, forecastHour);
  const weather = weatherPresentation(forecastWeatherCode);

  const option = useMemo(() => {
    const angle = (((forecastWindDirection ?? 90) - 90) * Math.PI) / 180;
    const windLength = 0.12 + Math.min(forecastWind ?? 0, 24) * 0.006;
    const windColor =
      (forecastGust ?? forecastWind ?? 0) >= 18
        ? pf.functional.red
        : (forecastGust ?? forecastWind ?? 0) >= 11
          ? '#FFB547'
          : '#FFD35A';
    const windStreams = [
      [-5.84, 35.84],
      [-5.7, 35.95],
      [-5.56, 36.06],
      [-5.42, 35.85],
      [-5.25, 35.96],
    ].map(([longitude, latitude], index) => ({
      coords: [
        [longitude, latitude],
        [longitude + Math.cos(angle) * windLength, latitude + Math.sin(angle) * windLength],
      ],
      lineStyle: { opacity: 0.55 + index * 0.07 },
      tooltip: `<b>Vent ${cardinal(forecastWindDirection)}</b><br/>${format(forecastWind)} m/s · rafales ${format(forecastGust)} m/s`,
    }));
    const impactPoints = impacts.slice(0, 12).map((impact, index) => ({
      name: impact.vessel_name ?? 'Navire non identifié',
      value: [...vesselPosition(impact.port_call_id, index), impact.combined_priority_score * 100],
      symbol: 'triangle',
      symbolRotate: index % 2 === 0 ? 74 : 252,
      symbolSize: 10 + impact.combined_priority_score * 10,
      itemStyle: {
        color:
          impact.combined_priority_score >= 0.65
            ? pf.functional.red
            : impact.combined_priority_score >= 0.4
              ? pf.functional.amber
              : pf.functional.green,
        shadowBlur: 14,
        shadowColor: 'rgba(0,0,0,.45)',
      },
      tooltip: `<b>${impact.vessel_name ?? 'Navire non identifié'}</b><br/>${impact.terminal_code ?? impact.port_code ?? 'Terminal à confirmer'}<br/>Priorité ${Math.round(impact.combined_priority_score * 100)} / 100`,
    }));
    const rainIntensity = Math.max(8, forecastRain ?? 0);
    const rainCells = [
      [-5.78, 36.05, 0.82],
      [-5.6, 35.88, 1],
      [-5.39, 36.03, 0.68],
      [-5.2, 35.88, 0.52],
    ].map(([longitude, latitude, scale], index) => ({
      name: `Cellule pluie ${index + 1}`,
      value: [longitude, latitude, rainIntensity * scale],
      symbolSize: 18 + rainIntensity * scale * 0.48,
      itemStyle: {
        color: index % 2 ? 'rgba(73,167,255,.42)' : 'rgba(141,145,255,.48)',
        borderColor: 'rgba(200,225,255,.58)',
        borderWidth: 1,
      },
      tooltip: `<b>Probabilité de pluie</b><br/>${format(forecastRain, 0)} % sur le secteur`,
    }));
    const wavePoints = [
      [-5.91, 35.82],
      [-5.76, 35.86],
      [-5.62, 35.9],
      [-5.35, 35.99],
      [-5.18, 36.03],
    ].map(([longitude, latitude], index) => ({
      name: `État de mer ${index + 1}`,
      value: [longitude, latitude, forecastWave ?? 0],
      symbolSize: 8 + Math.min(forecastWave ?? 0, 4) * 5,
      tooltip: `<b>État de mer</b><br/>Vague totale ${format(forecastWave, 2)} m<br/>Houle ${format(forecastSwell, 2)} m · période ${format(forecastPeriod)} s`,
    }));

    return {
      backgroundColor: '#082F4B',
      animationDuration: 1050,
      animationDurationUpdate: 600,
      animationEasing: 'cubicOut',
      animationEasingUpdate: 'cubicInOut',
      tooltip: {
        trigger: 'item',
        confine: true,
        padding: [12, 14],
        backgroundColor: 'rgba(13,15,18,.97)',
        borderColor: activeLayer.color,
        borderWidth: 1,
        extraCssText: 'border-radius:12px;box-shadow:0 18px 48px rgba(0,0,0,.38);',
        textStyle: { color: pf.text.primary, fontFamily: 'Inter, Segoe UI, sans-serif' },
        formatter: (params: { data?: { tooltip?: string }; name?: string }) =>
          params.data?.tooltip ?? params.name ?? '',
      },
      geo: {
        map: 'portflow-strait',
        center: activeZone.center,
        zoom: activeZone.zoom,
        roam: true,
        scaleLimit: { min: 10, max: 46 },
        label: { show: false },
        itemStyle: {
          areaColor: '#66533F',
          borderColor: '#B99A6C',
          borderWidth: 0.85,
          shadowBlur: 9,
          shadowColor: 'rgba(0,0,0,.22)',
        },
        regions: [
          { name: 'Morocco', itemStyle: { areaColor: '#705A42' } },
          { name: 'Spain', itemStyle: { areaColor: '#897052' } },
          { name: 'Portugal', itemStyle: { areaColor: '#7C664B' } },
        ],
        emphasis: { itemStyle: { areaColor: '#A1835C' }, label: { show: false } },
      },
      series: [
        {
          name: 'Routes d’approche',
          type: 'lines',
          coordinateSystem: 'geo',
          polyline: true,
          lineStyle: {
            color: layer === 'VESSELS' ? activeLayer.color : '#FFC44D',
            width: layer === 'VESSELS' ? 3 : 1.8,
            opacity: layer === 'VESSELS' ? 0.92 : 0.48,
            curveness: 0.1,
          },
          effect: {
            show: true,
            period: layer === 'VESSELS' ? 2.7 : 5.2,
            trailLength: 0.14,
            symbol: 'arrow',
            symbolSize: 7,
            color: layer === 'VESSELS' ? activeLayer.color : '#FFD35A',
          },
          data: [
            {
              coords: [[-6.05, 35.82], [-5.78, 35.86], TANGER_MED],
              tooltip: '<b>Approche Atlantique</b><br/>Route vers Tanger Med',
            },
            {
              coords: [[-5.08, 36.04], [-5.29, 35.99], TANGER_MED],
              tooltip: '<b>Approche Méditerranée</b><br/>Route vers Tanger Med',
            },
            {
              coords: [TANGER_MED, ALGECIRAS],
              tooltip: '<b>Couloir du détroit</b><br/>Tanger Med – Algésiras',
            },
          ],
        },
        {
          name: 'Flux de vent',
          type: 'lines',
          coordinateSystem: 'geo',
          lineStyle: {
            color: windColor,
            width: layer === 'WIND' ? 2.2 : 0,
            opacity: layer === 'WIND' ? 0.82 : 0,
            curveness: 0.14,
          },
          effect: {
            show: layer === 'WIND',
            period: Math.max(1.4, 3.2 - (forecastWind ?? 0) / 12),
            trailLength: 0.42,
            symbol: 'arrow',
            symbolSize: 10,
            color: '#FFF2A8',
          },
          data: windStreams,
        },
        {
          name: 'État de mer',
          type: 'effectScatter',
          coordinateSystem: 'geo',
          data: layer === 'WAVES' ? wavePoints : [],
          symbol: 'circle',
          rippleEffect: {
            scale: 2.4 + Math.min(forecastWave ?? 0, 3),
            brushType: 'stroke',
            period: 3.5,
          },
          itemStyle: { color: '#35D7F2', shadowBlur: 14, shadowColor: '#35D7F2' },
        },
        {
          name: 'Cellules de pluie',
          type: 'effectScatter',
          coordinateSystem: 'geo',
          data: layer === 'RAIN' ? rainCells : [],
          rippleEffect: { scale: 2.2, brushType: 'fill', period: 4 },
        },
        {
          name: 'Tanger Med',
          type: 'effectScatter',
          coordinateSystem: 'geo',
          rippleEffect: { scale: 3.5, brushType: 'stroke', period: 3 },
          symbolSize: 18,
          itemStyle: { color: activeLayer.color, shadowBlur: 18, shadowColor: activeLayer.color },
          label: {
            show: true,
            position: 'bottom',
            formatter: 'TANGER MED',
            color: '#FFFFFF',
            fontFamily: 'Inter, sans-serif',
            fontWeight: 800,
            fontSize: 10,
          },
          data: [
            {
              name: 'Tanger Med',
              value: [...TANGER_MED, forecastWave ?? 0],
              tooltip: `<b>Tanger Med · ${isForecast ? `prévision H+${forecastHour}` : 'observation actuelle'}</b><br/>${weather.label} · ${format(forecastTemperature)} °C<br/>Vent ${cardinal(forecastWindDirection)} ${format(forecastWind)} m/s<br/>Vague ${format(forecastWave, 2)} m · houle ${format(forecastSwell, 2)} m<br/>Période ${format(forecastPeriod)} s · pluie ${format(forecastRain, 0)} %`,
            },
          ],
        },
        {
          name: 'Bouée météo-marine',
          type: 'scatter',
          coordinateSystem: 'geo',
          symbol: 'pin',
          symbolSize: layer === 'WAVES' ? 36 : 27,
          itemStyle: { color: layer === 'WAVES' ? '#35D7F2' : '#49A7FF' },
          data: [
            {
              name: 'Point côtier',
              value: [-5.59, 35.89, forecastWave ?? 0],
              tooltip: `<b>Point météo-marin</b><br/>Vague totale ${format(forecastWave, 2)} m<br/>Houle ${format(forecastSwell, 2)} m<br/>Période dominante ${format(forecastPeriod)} s`,
            },
          ],
        },
        {
          name: 'Navires exposés',
          type: 'scatter',
          coordinateSystem: 'geo',
          data: layer === 'VESSELS' ? impactPoints : [],
          emphasis: { scale: 1.65, itemStyle: { borderColor: '#fff', borderWidth: 1 } },
        },
        {
          name: 'Algésiras',
          type: 'scatter',
          coordinateSystem: 'geo',
          symbolSize: 9,
          itemStyle: { color: '#7AE5A9' },
          label: {
            show: true,
            position: 'right',
            formatter: 'ALGÉSIRAS',
            color: '#D5E7EB',
            fontSize: 9,
          },
          data: [{ name: 'Algésiras', value: ALGECIRAS }],
        },
      ],
    };
  }, [
    activeLayer.color,
    activeZone.center,
    activeZone.zoom,
    forecastGust,
    forecastHour,
    forecastPeriod,
    forecastRain,
    forecastSwell,
    forecastTemperature,
    forecastWave,
    forecastWind,
    forecastWindDirection,
    impacts,
    isForecast,
    layer,
    weather.label,
  ]);

  const layerOverlay = {
    WAVES:
      'radial-gradient(ellipse at 44% 60%, rgba(53,215,242,.23), transparent 18%), repeating-radial-gradient(ellipse at 44% 60%, transparent 0 27px, rgba(135,231,255,.075) 29px 31px)',
    WIND: `repeating-linear-gradient(${forecastWindDirection ?? 90}deg, transparent 0 42px, rgba(255,211,90,.075) 44px 46px, transparent 48px 82px)`,
    RAIN: 'radial-gradient(circle at 38% 34%, rgba(141,145,255,.25), transparent 23%), radial-gradient(circle at 68% 56%, rgba(73,167,255,.2), transparent 20%)',
    VESSELS: 'radial-gradient(circle at 48% 64%, rgba(84,227,178,.14), transparent 31%)',
  }[layer];

  const layerValue = {
    WAVES: `${format(forecastWave, 2)} m`,
    WIND: `${format(forecastWind)} m/s`,
    RAIN: `${format(forecastRain, 0)} %`,
    VESSELS: `${impacts.length} navires`,
  }[layer];

  const metrics = [
    {
      label: 'Vague totale',
      value: `${format(forecastWave, 2)} m`,
      icon: 'lucide:waves',
      layer: 'WAVES',
    },
    {
      label: 'Houle',
      value: `${format(forecastSwell, 2)} m`,
      icon: 'lucide:activity',
      layer: 'WAVES',
    },
    {
      label: 'Période',
      value: `${format(forecastPeriod)} s`,
      icon: 'lucide:timer',
      layer: 'WAVES',
    },
    {
      label: 'Vent',
      value: `${cardinal(forecastWindDirection)} · ${format(forecastWind)} m/s`,
      icon: 'lucide:wind',
      layer: 'WIND',
    },
    {
      label: 'Pluie',
      value: `${format(forecastRain, 0)} %`,
      icon: 'lucide:droplets',
      layer: 'RAIN',
    },
  ] as const;

  return (
    <Box
      position="relative"
      sx={{
        minHeight: height,
        bgcolor: '#082F4B',
        overflow: 'hidden',
        isolation: 'isolate',
        '@keyframes mapWeatherFloat': {
          '0%,100%': { transform: 'translate3d(0,0,0) scale(1)' },
          '50%': { transform: 'translate3d(-10px,8px,0) scale(1.04)' },
        },
        '@keyframes weatherIconPulse': {
          '0%,100%': { filter: `drop-shadow(0 0 10px ${weather.color}55)` },
          '50%': { filter: `drop-shadow(0 0 24px ${weather.color}AA)` },
        },
      }}
    >
      <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />
      <Box
        aria-hidden="true"
        sx={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          background: layerOverlay,
          transition: 'background 520ms ease',
          zIndex: 1,
        }}
      />
      <Box
        aria-hidden="true"
        sx={{
          position: 'absolute',
          width: 360,
          height: 360,
          top: -210,
          right: -80,
          borderRadius: '50%',
          background: `radial-gradient(circle, ${weather.glow} 0%, transparent 70%)`,
          animation: 'mapWeatherFloat 8s ease-in-out infinite',
          pointerEvents: 'none',
          zIndex: 1,
        }}
      />

      <Stack
        direction={{ xs: 'column', md: 'row' }}
        gap={0.9}
        sx={{ position: 'absolute', top: 16, left: 16, right: 16, zIndex: 3 }}
      >
        <Stack direction="row" gap={0.8} alignItems="stretch">
          <Box
            sx={{
              width: 48,
              minHeight: 48,
              display: 'grid',
              placeItems: 'center',
              borderRadius: '14px',
              color: weather.color,
              bgcolor: 'rgba(13,15,18,.9)',
              border: `1px solid ${weather.color}55`,
              boxShadow: `0 14px 34px ${weather.glow}`,
              backdropFilter: 'blur(16px)',
            }}
          >
            <IconifyIcon
              icon={weather.icon}
              sx={{ fontSize: 27, animation: 'weatherIconPulse 2.8s ease-in-out infinite' }}
            />
          </Box>
          <Box
            sx={{
              px: 1.35,
              py: 0.72,
              minWidth: { xs: 0, sm: 220 },
              borderRadius: '14px',
              bgcolor: 'rgba(13,15,18,.92)',
              border: `1px solid ${isForecast ? '#8D91FF66' : '#54E3B266'}`,
              boxShadow: '0 16px 42px rgba(0,0,0,.3)',
              backdropFilter: 'blur(16px)',
            }}
          >
            <Stack direction="row" alignItems="center" gap={0.7}>
              <Box
                sx={{
                  width: 7,
                  height: 7,
                  borderRadius: '50%',
                  bgcolor: isForecast ? '#8D91FF' : '#54E3B2',
                  boxShadow: `0 0 12px ${isForecast ? '#8D91FF' : '#54E3B2'}`,
                }}
              />
              <Typography
                sx={{ color: isForecast ? '#AEB0FF' : '#7AF2C1', fontSize: 9, fontWeight: 900 }}
              >
                {isForecast ? `PRÉVISION · H+${forecastHour}` : 'OBSERVATION · MAINTENANT'}
              </Typography>
            </Stack>
            <Stack direction="row" alignItems="baseline" gap={0.65} mt={0.2}>
              <Typography sx={{ color: '#F5FBFC', fontSize: 16, fontWeight: 800 }}>
                {format(forecastTemperature)}°
              </Typography>
              <Typography sx={{ color: weather.color, fontSize: 10.5, fontWeight: 700 }}>
                {weather.label}
              </Typography>
            </Stack>
          </Box>
        </Stack>

        <Stack
          direction="row"
          flexWrap="wrap"
          gap={0.45}
          sx={{
            ml: { md: 'auto' },
            p: 0.45,
            alignSelf: { xs: 'flex-start', md: 'center' },
            bgcolor: 'rgba(13,15,18,.92)',
            border: '1px solid rgba(213,206,190,.16)',
            borderRadius: '14px',
            boxShadow: '0 16px 42px rgba(0,0,0,.3)',
            backdropFilter: 'blur(16px)',
          }}
        >
          {(Object.keys(layerDefinitions) as MapLayer[]).map((value) => {
            const definition = layerDefinitions[value];
            const selected = layer === value;
            return (
              <Button
                key={value}
                aria-pressed={selected}
                aria-label={`Couche ${definition.label}`}
                onClick={() => setLayer(value)}
                startIcon={<IconifyIcon icon={definition.icon} sx={{ fontSize: 15 }} />}
                sx={{
                  minWidth: 0,
                  minHeight: 38,
                  px: 1,
                  color: selected ? '#0B0D10' : pf.text.secondary,
                  bgcolor: selected ? definition.color : 'transparent',
                  border: `1px solid ${selected ? definition.color : 'transparent'}`,
                  borderRadius: '10px',
                  fontSize: 9.5,
                  boxShadow: selected ? `0 7px 18px ${definition.soft}` : 'none',
                  transition: 'all 220ms ease',
                  '&:hover': {
                    color: selected ? '#0B0D10' : definition.color,
                    bgcolor: selected ? definition.color : definition.soft,
                  },
                }}
              >
                {definition.label}
              </Button>
            );
          })}
        </Stack>
      </Stack>

      <Box
        sx={{
          position: 'absolute',
          top: { xs: 136, md: 92 },
          left: 16,
          zIndex: 3,
          px: 1.1,
          py: 0.65,
          borderRadius: '11px',
          bgcolor: 'rgba(13,15,18,.88)',
          border: `1px solid ${activeLayer.color}44`,
          backdropFilter: 'blur(14px)',
        }}
      >
        <Typography sx={{ color: pf.text.secondary, fontSize: 8.5, fontWeight: 800 }}>
          {activeZone.label.toUpperCase()} · {activeLayer.label.toUpperCase()}
        </Typography>
        <Typography sx={{ color: activeLayer.color, fontSize: 15, fontWeight: 800 }}>
          {layerValue}
        </Typography>
      </Box>

      <Stack
        direction="row"
        gap={0.45}
        sx={{
          position: 'absolute',
          left: 16,
          bottom: 96,
          zIndex: 3,
          p: 0.42,
          bgcolor: 'rgba(13,15,18,.9)',
          border: '1px solid rgba(213,206,190,.15)',
          borderRadius: '12px',
          backdropFilter: 'blur(14px)',
          maxWidth: 'calc(100% - 32px)',
          overflowX: 'auto',
        }}
      >
        {(Object.keys(zoneDefinitions) as MapZone[]).map((value) => {
          const selected = zone === value;
          return (
            <Button
              key={value}
              aria-pressed={selected}
              onClick={() => setZone(value)}
              sx={{
                minWidth: 0,
                whiteSpace: 'nowrap',
                px: 1,
                minHeight: 30,
                borderRadius: '8px',
                color: selected ? '#0B0D10' : pf.text.secondary,
                bgcolor: selected ? '#E5B96B' : 'transparent',
                fontSize: 8.5,
                '&:hover': { bgcolor: selected ? '#F0C77E' : 'rgba(229,185,107,.12)' },
              }}
            >
              {zoneDefinitions[value].short}
            </Button>
          );
        })}
      </Stack>

      <Box
        sx={{
          position: 'absolute',
          left: 16,
          right: 16,
          bottom: 14,
          zIndex: 3,
          display: 'grid',
          gridTemplateColumns: { xs: 'repeat(2,minmax(0,1fr))', sm: 'repeat(5,minmax(0,1fr))' },
          gap: 0.45,
          p: 0.55,
          bgcolor: 'rgba(13,15,18,.93)',
          border: '1px solid rgba(213,206,190,.16)',
          borderRadius: '14px',
          boxShadow: '0 18px 48px rgba(0,0,0,.32)',
          backdropFilter: 'blur(18px)',
        }}
      >
        {metrics.map((metric) => {
          const selected = layer === metric.layer;
          const color = layerDefinitions[metric.layer].color;
          return (
            <Stack
              key={metric.label}
              direction="row"
              alignItems="center"
              gap={0.75}
              sx={{
                minWidth: 0,
                px: 0.85,
                py: 0.65,
                borderRadius: '9px',
                bgcolor: selected ? `${color}12` : 'transparent',
                border: `1px solid ${selected ? `${color}35` : 'transparent'}`,
                transition: 'all 220ms ease',
              }}
            >
              <IconifyIcon icon={metric.icon} sx={{ color, fontSize: 16, flexShrink: 0 }} />
              <Box minWidth={0}>
                <Typography noWrap sx={{ color: pf.text.tertiary, fontSize: 7.5, fontWeight: 800 }}>
                  {metric.label.toUpperCase()}
                </Typography>
                <Typography noWrap sx={{ color: '#F5FBFC', fontSize: 10.5, fontWeight: 750 }}>
                  {metric.value}
                </Typography>
              </Box>
            </Stack>
          );
        })}
      </Box>
    </Box>
  );
};

export default MetoceanSituationMap;
