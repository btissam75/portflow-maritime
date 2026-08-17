import { Box, Button, Stack, Typography } from '@mui/material';
import { useMemo, useState } from 'react';
import * as echarts from 'echarts/core';
import { GeoComponent, TooltipComponent } from 'echarts/components';
import { EffectScatterChart, LinesChart, ScatterChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import world from 'assets/json/world.json';
import IconifyIcon from 'components/base/IconifyIcon';
import ReactEchart from 'components/base/ReactEhart';
import type { TowerVessel } from 'types/controlTower';
import { portflowPalette as pf } from 'theme/portflowPalette';

echarts.use([
  GeoComponent,
  TooltipComponent,
  EffectScatterChart,
  LinesChart,
  ScatterChart,
  CanvasRenderer,
]);
// @ts-ignore Bundled GeoJSON is compatible with ECharts.
echarts.registerMap('tower-maritime', { geoJSON: world });

const PORT: [number, number] = [-5.501, 35.891];

const MaritimeApproachMap = ({
  vessels,
  selected,
  onSelect,
}: {
  vessels: TowerVessel[];
  selected?: string;
  onSelect: (id: string) => void;
}) => {
  const [layer, setLayer] = useState<'TRAFFIC' | 'ETA' | 'RISK'>('TRAFFIC');
  const option = useMemo(
    () => ({
      backgroundColor: pf.map.water,
      animationDuration: 1000,
      tooltip: {
        trigger: 'item',
        backgroundColor: 'rgba(13,15,18,.97)',
        borderColor: pf.functional.cyan,
        padding: [12, 14],
        extraCssText: 'border-radius:12px;box-shadow:0 18px 48px rgba(0,0,0,.42)',
        textStyle: { color: pf.text.primary },
        formatter: (params: { data?: { tooltip?: string }; name?: string }) =>
          params.data?.tooltip ?? params.name ?? '',
      },
      geo: {
        map: 'tower-maritime',
        center: [-6.15, 35.82],
        zoom: 43,
        roam: true,
        scaleLimit: { min: 20, max: 70 },
        label: { show: false },
        itemStyle: { areaColor: '#725E48', borderColor: '#B8996B', borderWidth: 0.8 },
        regions: [
          { name: 'Morocco', itemStyle: { areaColor: '#725E48' } },
          { name: 'Spain', itemStyle: { areaColor: '#8A704F' } },
        ],
        emphasis: { itemStyle: { areaColor: '#A48863' }, label: { show: false } },
      },
      series: [
        {
          name: 'Trajectoires prévues',
          type: 'lines',
          coordinateSystem: 'geo',
          lineStyle: {
            color: layer === 'ETA' ? pf.functional.purple : pf.functional.cyan,
            width: 1.6,
            opacity: 0.45,
            curveness: 0.18,
          },
          effect: {
            show: true,
            period: 4,
            trailLength: 0.18,
            symbol: 'arrow',
            symbolSize: 7,
            color: layer === 'ETA' ? pf.functional.purple : pf.functional.cyan,
          },
          data: vessels.map((vessel) => ({
            coords: [[vessel.longitude, vessel.latitude], PORT],
            tooltip: `<b>${vessel.name}</b><br/>Trajectoire prévue vers Tanger Med`,
          })),
        },
        {
          name: 'Navires',
          type: 'effectScatter',
          coordinateSystem: 'geo',
          symbol: 'triangle',
          showEffectOn: 'render',
          rippleEffect: { scale: 2.4, brushType: 'stroke', period: 4 },
          data: vessels.map((vessel) => ({
            name: vessel.name,
            vesselId: vessel.vessel_id,
            value: [vessel.longitude, vessel.latitude, vessel.congestion_risk * 100],
            symbolRotate: vessel.heading,
            symbolSize: selected === vessel.vessel_id ? 27 : 18,
            itemStyle: {
              color:
                layer === 'RISK'
                  ? vessel.congestion_risk >= 0.7
                    ? pf.functional.red
                    : vessel.congestion_risk >= 0.48
                      ? pf.functional.amber
                      : pf.functional.green
                  : selected === vessel.vessel_id
                    ? '#fff'
                    : pf.functional.cyan,
              borderColor: selected === vessel.vessel_id ? pf.functional.cyan : '#0A3150',
              borderWidth: 2,
              shadowBlur: selected === vessel.vessel_id ? 22 : 9,
              shadowColor: pf.functional.cyan,
            },
            tooltip: `<b>${vessel.name}</b><br/>${vessel.status} · ${vessel.speed_kn} nd<br/>Distance ${vessel.distance_nm} NM<br/>ETA recalculée ${new Date(vessel.predicted_eta).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}<br/>${vessel.associated_units} unités associées`,
          })),
        },
        {
          name: 'Tanger Med',
          type: 'effectScatter',
          coordinateSystem: 'geo',
          symbolSize: 17,
          rippleEffect: { scale: 3.8, brushType: 'stroke' },
          itemStyle: { color: pf.functional.amber },
          label: {
            show: true,
            formatter: 'TANGER MED',
            position: 'bottom',
            color: '#fff',
            fontSize: 10,
            fontWeight: 800,
          },
          data: [
            {
              name: 'Tanger Med',
              value: PORT,
              tooltip: '<b>Tanger Med</b><br/>Zone portuaire de destination',
            },
          ],
        },
      ],
    }),
    [layer, selected, vessels],
  );

  const onEvents = useMemo(
    () => ({
      click: (params: { data?: { vesselId?: string } }) => {
        if (params.data?.vesselId) onSelect(params.data.vesselId);
      },
    }),
    [onSelect],
  );
  return (
    <Box
      sx={{
        position: 'relative',
        overflow: 'hidden',
        borderRadius: '18px',
        border: `1px solid ${pf.structure.border}`,
      }}
    >
      <ReactEchart
        echarts={echarts}
        option={option}
        onEvents={onEvents}
        sx={{ width: 1, height: 540 }}
      />
      <Stack
        direction="row"
        gap={0.4}
        sx={{
          position: 'absolute',
          top: 14,
          right: 14,
          p: 0.4,
          bgcolor: 'rgba(13,15,18,.92)',
          border: `1px solid ${pf.structure.border}`,
          borderRadius: '12px',
          backdropFilter: 'blur(14px)',
        }}
      >
        {(
          [
            ['TRAFFIC', 'Trafic', 'lucide:ship'],
            ['ETA', 'ETA', 'lucide:clock-3'],
            ['RISK', 'Risque', 'lucide:shield-alert'],
          ] as const
        ).map(([value, label, icon]) => (
          <Button
            key={value}
            aria-pressed={layer === value}
            onClick={() => setLayer(value)}
            startIcon={<IconifyIcon icon={icon} />}
            sx={{
              color: layer === value ? pf.background.primary : pf.text.secondary,
              bgcolor: layer === value ? pf.functional.cyan : 'transparent',
              borderRadius: '9px',
              fontSize: 9,
            }}
          >
            {label}
          </Button>
        ))}
      </Stack>
      <Typography
        sx={{
          position: 'absolute',
          left: 14,
          bottom: 12,
          px: 1,
          py: 0.55,
          color: pf.text.secondary,
          bgcolor: 'rgba(13,15,18,.88)',
          borderRadius: '9px',
          fontSize: 8.5,
        }}
      >
        Positions d’exercice · contrat AIS prêt pour raccordement
      </Typography>
    </Box>
  );
};

export default MaritimeApproachMap;
