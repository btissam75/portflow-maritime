import { Box, Stack, Typography } from '@mui/material';
import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import {
  GeoComponent,
  LegendComponent,
  TooltipComponent,
} from 'echarts/components';
import {
  EffectScatterChart,
  LinesChart,
  ScatterChart,
} from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import world from 'assets/json/world.json';
import ReactEchart from 'components/base/ReactEhart';
import type { LiveAtmosphereCurrent, LiveMarineCurrent } from 'types/liveMetocean';
import { portflowPalette as pf } from 'theme/portflowPalette';

echarts.use([
  TooltipComponent,
  LegendComponent,
  GeoComponent,
  LinesChart,
  ScatterChart,
  EffectScatterChart,
  CanvasRenderer,
]);

// @ts-ignore The bundled GeoJSON follows ECharts' map schema.
echarts.registerMap('portflow-strait', { geoJSON: world });

const TANGER_MED: [number, number] = [-5.501, 35.891];
const ALGECIRAS: [number, number] = [-5.448, 36.132];

const format = (value: number | null | undefined, digits = 1) =>
  value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits);

interface MetoceanSituationMapProps {
  atmosphere?: LiveAtmosphereCurrent;
  marine?: LiveMarineCurrent;
  height?: number;
}

const MetoceanSituationMap = ({ atmosphere, marine, height = 390 }: MetoceanSituationMapProps) => {
  const option = useMemo(
    () => ({
      backgroundColor: pf.map.water,
      animationDuration: 1100,
      animationEasing: 'cubicOut',
      tooltip: {
        trigger: 'item',
        confine: true,
        backgroundColor: pf.background.navigation,
        borderColor: pf.map.road,
        borderWidth: 1,
        textStyle: { color: pf.text.primary, fontFamily: 'Inter, Segoe UI, sans-serif' },
        formatter: (params: { data?: { tooltip?: string }; name?: string }) =>
          params.data?.tooltip ?? params.name ?? '',
      },
      legend: {
        left: 16,
        bottom: 14,
        itemWidth: 15,
        itemHeight: 7,
        textStyle: { color: pf.map.label, fontSize: 10 },
        data: ['Routes d’approche', 'Tanger Med', 'Point météo-marin'],
      },
      geo: {
        map: 'portflow-strait',
        center: [-5.53, 35.98],
        zoom: 19,
        roam: true,
        scaleLimit: { min: 11, max: 45 },
        label: { show: false },
        itemStyle: {
          areaColor: pf.map.terrain,
          borderColor: pf.map.outline,
          borderWidth: 0.8,
        },
        regions: [
          { name: 'Morocco', itemStyle: { areaColor: '#254A43' } },
          { name: 'Spain', itemStyle: { areaColor: '#3C5147' } },
          { name: 'Portugal', itemStyle: { areaColor: '#425047' } },
        ],
        emphasis: {
          itemStyle: { areaColor: pf.map.portZone },
          label: { show: false },
        },
      },
      series: [
        {
          name: 'Routes d’approche',
          type: 'lines',
          coordinateSystem: 'geo',
          polyline: true,
          lineStyle: { color: pf.functional.amber, width: 2.4, opacity: 0.82, curveness: 0.1 },
          effect: {
            show: true,
            period: 5,
            trailLength: 0.18,
            symbol: 'arrow',
            symbolSize: 7,
            color: pf.functional.amber,
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
          name: 'Tanger Med',
          type: 'effectScatter',
          coordinateSystem: 'geo',
          rippleEffect: { scale: 3.4, brushType: 'stroke' },
          symbolSize: 17,
          itemStyle: { color: pf.functional.cyan },
          label: {
            show: true,
            position: 'bottom',
            formatter: 'TANGER MED',
            color: pf.text.primary,
            fontFamily: 'Inter, sans-serif',
            fontWeight: 700,
            fontSize: 10,
          },
          data: [
            {
              name: 'Tanger Med',
              value: [...TANGER_MED, marine?.wave_height ?? 0],
              tooltip: `<b>Tanger Med</b><br/>Air ${format(atmosphere?.temperature_2m)} °C<br/>Vent ${format(atmosphere?.wind_speed_10m)} m/s<br/>Vague ${format(marine?.wave_height, 2)} m`,
            },
          ],
        },
        {
          name: 'Point météo-marin',
          type: 'scatter',
          coordinateSystem: 'geo',
          symbol: 'pin',
          symbolSize: 28,
          itemStyle: { color: pf.functional.blue },
          data: [
            {
              name: 'Observation côtière',
              value: [-5.59, 35.89, marine?.wave_height ?? 0],
              tooltip: `<b>Observation côtière</b><br/>Houle ${format(marine?.swell_wave_height, 2)} m<br/>Période ${format(marine?.wave_period)} s<br/>Température mer ${format(marine?.sea_surface_temperature)} °C`,
            },
          ],
        },
        {
          name: 'Algésiras',
          type: 'scatter',
          coordinateSystem: 'geo',
          symbolSize: 8,
          itemStyle: { color: pf.functional.green },
          label: {
            show: true,
            position: 'right',
            formatter: 'ALGÉSIRAS',
            color: pf.map.label,
            fontSize: 9,
          },
          data: [{ name: 'Algésiras', value: ALGECIRAS }],
        },
      ],
    }),
    [atmosphere, marine],
  );

  return (
    <Box position="relative" sx={{ minHeight: height, bgcolor: pf.map.water }}>
      <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />
      <Stack
        direction="row"
        gap={0.75}
        sx={{ position: 'absolute', top: 14, left: 14, pointerEvents: 'none' }}
      >
        <Box sx={{ px: 1.1, py: 0.65, borderRadius: '5px', bgcolor: 'rgba(7,23,34,0.92)', border: `1px solid ${pf.structure.border}` }}>
          <Typography sx={{ color: pf.text.primary, fontFamily: 'Inter, sans-serif', fontSize: 10, fontWeight: 600 }}>
            DÉTROIT DE GIBRALTAR
          </Typography>
          <Typography sx={{ color: pf.text.secondary, fontSize: 10 }}>Glisser pour explorer · molette pour zoomer</Typography>
        </Box>
      </Stack>
    </Box>
  );
};

export default MetoceanSituationMap;
