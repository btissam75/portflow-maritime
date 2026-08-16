import { Box, Chip, Stack, useTheme } from '@mui/material';
import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import {
  GeoComponent,
  GeoComponentOption,
  LegendComponent,
  LegendComponentOption,
  TooltipComponent,
  TooltipComponentOption,
} from 'echarts/components';
import {
  EffectScatterChart,
  EffectScatterSeriesOption,
  LinesChart,
  LinesSeriesOption,
  ScatterChart,
  ScatterSeriesOption,
} from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import world from 'assets/json/world.json';
import ReactEchart from 'components/base/ReactEhart';
import IconifyIcon from 'components/base/IconifyIcon';
import type { OperationalSummary, WeatherPoint } from 'types/replay';

echarts.use([
  TooltipComponent,
  LegendComponent,
  GeoComponent,
  LinesChart,
  ScatterChart,
  EffectScatterChart,
  CanvasRenderer,
]);
// @ts-ignore The bundled map follows ECharts GeoJSON but has no generated TS declaration.
echarts.registerMap('portflow-world', { geoJSON: world });

type EChartsOption = echarts.ComposeOption<
  | TooltipComponentOption
  | LegendComponentOption
  | GeoComponentOption
  | LinesSeriesOption
  | ScatterSeriesOption
  | EffectScatterSeriesOption
>;

interface MapTooltipParams {
  name?: string;
  data?: {
    tooltip?: string;
  };
}

const formatMapTooltip = (params: unknown) => {
  if (params == null || typeof params !== 'object' || Array.isArray(params)) return '';
  const item = params as MapTooltipParams;
  return item.data?.tooltip ?? item.name ?? '';
};

interface OperationalMapProps {
  summary: OperationalSummary | null;
  weather: WeatherPoint[];
  height?: number;
  showCorridors?: boolean;
  variant?: 'light' | 'dark';
}

const OperationalMap = ({
  summary,
  weather,
  height = 440,
  showCorridors = true,
  variant = 'light',
}: OperationalMapProps) => {
  const theme = useTheme();
  const latestWeather = weather.length > 0 ? weather[weather.length - 1] : undefined;

  const option = useMemo<EChartsOption>(
    () => ({
      backgroundColor: variant === 'dark' ? '#102B34' : 'transparent',
      animationDuration: 700,
      tooltip: {
        trigger: 'item',
        backgroundColor: '#102A43',
        borderWidth: 0,
        textStyle: { color: '#FFFFFF' },
        formatter: formatMapTooltip,
      },
      legend: {
        left: 16,
        bottom: 14,
        itemWidth: 14,
        textStyle: { color: variant === 'dark' ? '#9AB4BA' : theme.palette.text.secondary },
        data: ['Corridors', 'Tanger Med', 'Météo'],
      },
      geo: {
        map: 'portflow-world',
        center: [-5.52, 35.94],
        zoom: 22,
        roam: true,
        scaleLimit: { min: 10, max: 55 },
        label: { show: false },
        itemStyle: {
          areaColor: variant === 'dark' ? '#1B3C45' : '#DDE8E7',
          borderColor: variant === 'dark' ? '#45636B' : '#91AAA9',
          borderWidth: 0.55,
        },
        emphasis: {
          itemStyle: { areaColor: variant === 'dark' ? '#294D56' : '#C8DBD8' },
          label: { show: false },
        },
      },
      series: [
        {
          name: 'Corridors',
          type: 'lines',
          coordinateSystem: 'geo',
          polyline: true,
          silent: false,
          lineStyle: {
            color: variant === 'dark' ? '#6FE0D3' : '#167D85',
            width: 2.4,
            opacity: variant === 'dark' ? 0.78 : 0.65,
            curveness: 0.12,
          },
          effect: {
            show: true,
            period: 5,
            trailLength: 0.25,
            symbol: 'arrow',
            symbolSize: 7,
            color: '#F3B64C',
          },
          data: showCorridors
            ? [
            {
              coords: [
                [-6.05, 35.82],
                [-5.82, 35.88],
                [-5.5, 35.8892],
              ],
              tooltip: 'Approche Atlantique vers Tanger Med',
            },
            {
              coords: [
                [-5.05, 36.03],
                [-5.26, 36.0],
                [-5.5, 35.8892],
              ],
              tooltip: 'Approche Méditerranée vers Tanger Med',
            },
            {
              coords: [
                [-5.5, 35.8892],
                [-5.44, 36.13],
              ],
              tooltip: 'Couloir Tanger Med - Algésiras',
            },
              ]
            : [],
        },
        {
          name: 'Tanger Med',
          type: 'effectScatter',
          coordinateSystem: 'geo',
          rippleEffect: { scale: 3.2, brushType: 'stroke' },
          symbolSize: 18,
          itemStyle: { color: '#6FE0D3', shadowBlur: 14, shadowColor: '#6FE0D3' },
          label: {
            show: true,
            position: 'bottom',
            formatter: 'TANGER MED',
            color: variant === 'dark' ? '#F4FAFA' : '#102A43',
            fontWeight: 700,
            fontSize: 11,
          },
          data: [
            {
              name: 'Tanger Med',
              value: [-5.5, 35.8892, summary?.expected_next_24h ?? 0],
              tooltip: `<strong>Tanger Med</strong><br/>${summary?.expected_next_24h ?? 0} escales attendues à 24 h<br/>${summary?.vessels_in_port ?? 0} navires à quai`,
            },
          ],
        },
        {
          name: 'Météo',
          type: 'scatter',
          coordinateSystem: 'geo',
          symbol: 'pin',
          symbolSize: 24,
          itemStyle: { color: '#D97706' },
          data: latestWeather
            ? [
                {
                  name: 'Point météo',
                  value: [
                    latestWeather.longitude,
                    latestWeather.latitude,
                    latestWeather.wave_height_m ?? 0,
                  ],
                  tooltip: `<strong>Observation maritime</strong><br/>Vagues ${latestWeather.wave_height_m?.toFixed(1) ?? '—'} m<br/>Vent ${latestWeather.wind_speed_ms?.toFixed(1) ?? '—'} m/s`,
                },
              ]
            : [],
        },
      ],
    }),
    [latestWeather, showCorridors, summary, theme, variant],
  );

  return (
    <Box position="relative" sx={{ bgcolor: variant === 'dark' ? '#102B34' : 'transparent' }}>
      <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />
      <Stack
        direction="row"
        gap={0.75}
        flexWrap="wrap"
        sx={{ position: 'absolute', top: 12, left: 12 }}
      >
        <Chip
          size="small"
          icon={<IconifyIcon icon="material-symbols:radar-rounded" />}
          label={
            summary?.ais_positions_72h
              ? `${summary.ais_vessels_72h} navires AIS`
              : 'AIS indisponible'
          }
          color={summary?.ais_positions_72h ? 'success' : 'error'}
          sx={{
            bgcolor: variant === 'dark' ? '#173A43' : 'background.paper',
            color: variant === 'dark' ? '#E7F2F3' : undefined,
            borderColor: variant === 'dark' ? '#49666D' : undefined,
          }}
        />
        <Chip
          size="small"
          label="Détroit de Gibraltar"
          variant="outlined"
          sx={{
            bgcolor: variant === 'dark' ? '#173A43' : 'background.paper',
            color: variant === 'dark' ? '#E7F2F3' : undefined,
            borderColor: variant === 'dark' ? '#49666D' : undefined,
          }}
        />
      </Stack>
    </Box>
  );
};

export default OperationalMap;
