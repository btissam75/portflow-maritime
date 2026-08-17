import { Box, Button, Stack, Typography } from '@mui/material';
import { useMemo, useState } from 'react';
import * as echarts from 'echarts/core';
import { GeoComponent, LegendComponent, TooltipComponent } from 'echarts/components';
import { EffectScatterChart, LinesChart, ScatterChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import world from 'assets/json/world.json';
import IconifyIcon from 'components/base/IconifyIcon';
import ReactEchart from 'components/base/ReactEhart';
import type { CapacityDecision } from 'types/capacity';
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
echarts.registerMap('portflow-capacity-strait', { geoJSON: world });

type MapLayer = 'TRAFFIC' | 'RISK' | 'CAPACITY';

const TANGER_MED: [number, number] = [-5.501, 35.891];

const hash = (value: string) =>
  [...value].reduce((total, character) => (total * 31 + character.charCodeAt(0)) >>> 0, 7);

const positionFor = (decision: CapacityDecision): [number, number] => {
  const seed = hash(decision.port_call_id);
  const dx = ((seed % 100) / 100 - 0.5) * 0.12;
  const dy = (((seed >> 7) % 100) / 100 - 0.5) * 0.075;
  if (decision.hsmm_state === 'BERTH_WINDOW') return [-5.505 + dx * 0.2, 35.888 + dy * 0.2];
  if (decision.hsmm_state === 'WAITING') return [-5.63 + dx * 0.55, 35.91 + dy * 0.6];
  return [-5.84 + dx, 35.84 + dy];
};

const riskColor = (risk: number) =>
  risk >= 0.65 ? pf.functional.red : risk >= 0.4 ? pf.functional.amber : pf.functional.green;

const CapacityOperationsMap = ({
  decisions,
  selectedId,
  onSelect,
  height = 430,
}: {
  decisions: CapacityDecision[];
  selectedId: string | null;
  onSelect: (portCallId: string) => void;
  height?: number;
}) => {
  const [layer, setLayer] = useState<MapLayer>('RISK');
  const selected = decisions.find((decision) => decision.port_call_id === selectedId);

  const option = useMemo(() => {
    const vessels = decisions.slice(0, 40).map((decision) => {
      const position = positionFor(decision);
      const color =
        layer === 'RISK'
          ? riskColor(decision.risk_score)
          : layer === 'CAPACITY'
            ? decision.watchlist_selected
              ? pf.functional.purple
              : pf.functional.cyan
            : pf.functional.blue;
      return {
        name: decision.vessel_name ?? 'Navire non identifié',
        portCallId: decision.port_call_id,
        value: [...position, decision.risk_score * 100],
        itemStyle: { color },
        symbolSize: decision.port_call_id === selectedId ? 17 : 8 + decision.risk_score * 6,
        tooltip: [
          `<b>${decision.vessel_name ?? 'Navire non identifié'}</b>`,
          `${decision.terminal_code ?? decision.port_code ?? 'Terminal à confirmer'}`,
          `Risque ${Math.round(decision.risk_score * 100)} %`,
          `Temps estimé ${decision.remaining_p50_h.toFixed(1)} h`,
          decision.watchlist_selected ? 'Revue recommandée' : 'Surveillance normale',
        ].join('<br/>'),
      };
    });
    const selectedPoint = vessels.filter((vessel) => vessel.portCallId === selectedId);
    const normalPoints = vessels.filter((vessel) => vessel.portCallId !== selectedId);

    return {
      backgroundColor: pf.map.water,
      animationDuration: 850,
      animationDurationUpdate: 500,
      animationEasingUpdate: 'cubicInOut',
      tooltip: {
        trigger: 'item',
        confine: true,
        backgroundColor: pf.background.navigation,
        borderColor: pf.map.road,
        textStyle: { color: pf.text.primary, fontFamily: 'Inter, sans-serif', fontSize: 11 },
        formatter: (params: { data?: { tooltip?: string }; name?: string }) =>
          params.data?.tooltip ?? params.name ?? '',
      },
      geo: {
        map: 'portflow-capacity-strait',
        center: [-5.63, 35.9],
        zoom: 23,
        roam: true,
        scaleLimit: { min: 14, max: 48 },
        label: { show: false },
        itemStyle: {
          areaColor: pf.map.terrain,
          borderColor: pf.map.outline,
          borderWidth: 0.75,
        },
        regions: [
          { name: 'Morocco', itemStyle: { areaColor: '#173B3C' } },
          { name: 'Spain', itemStyle: { areaColor: '#324D45' } },
        ],
        emphasis: { itemStyle: { areaColor: pf.map.portZone }, label: { show: false } },
      },
      series: [
        {
          name: 'Couloirs d’approche',
          type: 'lines',
          coordinateSystem: 'geo',
          polyline: true,
          silent: true,
          lineStyle: { color: pf.functional.cyan, width: 1.3, opacity: 0.28, curveness: 0.12 },
          effect: {
            show: true,
            period: layer === 'TRAFFIC' ? 3.5 : 5.5,
            trailLength: 0.2,
            symbol: 'arrow',
            symbolSize: 6,
            color: pf.functional.cyan,
          },
          data: [
            { coords: [[-6.02, 35.79], [-5.79, 35.84], [-5.63, 35.91], TANGER_MED] },
            { coords: [[-5.14, 36.0], [-5.35, 35.95], [-5.63, 35.91], TANGER_MED] },
          ],
        },
        {
          name: 'Zones opérationnelles',
          type: 'scatter',
          coordinateSystem: 'geo',
          silent: true,
          symbol: 'roundRect',
          symbolSize: [56, 22],
          label: { show: true, color: pf.text.primary, fontSize: 8, fontWeight: 700 },
          data: [
            {
              name: 'APPROCHE',
              value: [-5.84, 35.84],
              itemStyle: { color: 'rgba(73,167,255,0.34)' },
              label: { formatter: 'APPROCHE' },
            },
            {
              name: 'ATTENTE',
              value: [-5.63, 35.91],
              itemStyle: { color: 'rgba(255,189,74,0.34)' },
              label: { formatter: 'ATTENTE' },
            },
            {
              name: 'TERMINAL',
              value: TANGER_MED,
              itemStyle: { color: 'rgba(54,214,207,0.38)' },
              label: { formatter: 'TERMINAL' },
            },
          ],
        },
        {
          name: 'Escales',
          type: 'scatter',
          coordinateSystem: 'geo',
          data: normalPoints,
          emphasis: { scale: 1.5, itemStyle: { borderColor: '#fff', borderWidth: 1 } },
        },
        {
          name: 'Escale sélectionnée',
          type: 'effectScatter',
          coordinateSystem: 'geo',
          rippleEffect: { scale: 3.5, brushType: 'stroke' },
          data: selectedPoint,
          label: {
            show: true,
            position: 'top',
            formatter: (params: { name: string }) => params.name.toUpperCase(),
            color: pf.text.primary,
            fontSize: 9,
            fontWeight: 700,
          },
        },
      ],
    };
  }, [decisions, layer, selectedId]);

  const events = useMemo(
    () => ({
      click: (params: { data?: { portCallId?: string } }) => {
        if (params.data?.portCallId) onSelect(params.data.portCallId);
      },
    }),
    [onSelect],
  );

  return (
    <Box position="relative" sx={{ minHeight: height, bgcolor: pf.map.water, overflow: 'hidden' }}>
      <ReactEchart echarts={echarts} option={option} onEvents={events} sx={{ width: 1, height }} />
      <Stack
        direction="row"
        gap={0.55}
        sx={{ position: 'absolute', top: 14, left: 14, right: 14, alignItems: 'flex-start' }}
      >
        <Box
          sx={{
            px: 1.1,
            py: 0.75,
            color: pf.text.primary,
            bgcolor: 'rgba(13,15,18,.92)',
            border: `1px solid ${pf.structure.border}`,
            borderRadius: '7px',
            backdropFilter: 'blur(10px)',
          }}
        >
          <Stack direction="row" alignItems="center" gap={0.65}>
            <Box
              sx={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                bgcolor: selected ? riskColor(selected.risk_score) : pf.functional.cyan,
                animation: 'portflowPulse 2s infinite',
              }}
            />
            <Typography sx={{ fontSize: 10, fontWeight: 750 }}>
              {selected?.vessel_name?.toUpperCase() ?? 'CARTE DES ESCALES'}
            </Typography>
          </Stack>
          <Typography sx={{ color: pf.text.tertiary, fontSize: 9, mt: 0.25 }}>
            Vue de processus · position indicative
          </Typography>
        </Box>
        <Stack
          direction="row"
          gap={0.35}
          sx={{
            ml: 'auto',
            p: 0.35,
            bgcolor: 'rgba(13,15,18,.92)',
            border: `1px solid ${pf.structure.border}`,
            borderRadius: '7px',
            backdropFilter: 'blur(10px)',
          }}
        >
          {(
            [
              ['TRAFFIC', 'Trafic', 'lucide:route'],
              ['RISK', 'Risque', 'lucide:shield-alert'],
              ['CAPACITY', 'Revues', 'lucide:list-checks'],
            ] as const
          ).map(([value, label, icon]) => (
            <Button
              key={value}
              aria-pressed={layer === value}
              aria-label={`Couche ${label}`}
              onClick={() => setLayer(value)}
              startIcon={<IconifyIcon icon={icon} sx={{ fontSize: 14 }} />}
              sx={{
                minWidth: 0,
                minHeight: 34,
                px: 0.9,
                color: layer === value ? pf.background.primary : pf.text.secondary,
                bgcolor: layer === value ? pf.functional.cyan : 'transparent',
                borderRadius: '5px',
                fontSize: 9.5,
                '&:hover': {
                  bgcolor: layer === value ? '#5CE4DE' : pf.background.panelHover,
                },
              }}
            >
              <Box component="span" sx={{ display: { xs: 'none', sm: 'inline' } }}>
                {label}
              </Box>
            </Button>
          ))}
        </Stack>
      </Stack>
      <Typography
        sx={{
          position: 'absolute',
          right: 14,
          bottom: 12,
          px: 0.8,
          py: 0.35,
          color: pf.text.tertiary,
          bgcolor: 'rgba(13,15,18,.86)',
          borderRadius: '4px',
          fontSize: 8.5,
          pointerEvents: 'none',
        }}
      >
        Cliquer un navire pour ouvrir sa fiche · glisser pour explorer
      </Typography>
    </Box>
  );
};

export default CapacityOperationsMap;
