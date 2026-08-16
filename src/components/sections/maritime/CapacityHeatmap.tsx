import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import { GridComponent, TooltipComponent, VisualMapComponent } from 'echarts/components';
import { HeatmapChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import type { CapacityDecision } from 'types/capacity';

echarts.use([TooltipComponent, VisualMapComponent, GridComponent, HeatmapChart, CanvasRenderer]);

const average = (values: number[]) => values.length
  ? values.reduce((total, value) => total + value, 0) / values.length
  : 0;

const formatPercent = (value: number) => value > 0 && value < 0.1
  ? '< 0,1 %'
  : `${value.toLocaleString('fr-FR', { maximumFractionDigits: value < 10 ? 1 : 0 })} %`;

const CapacityHeatmap = ({ decisions, height = 310 }: { decisions: CapacityDecision[]; height?: number }) => {
  const matrix = useMemo(() => {
    const grouped = new Map<string, CapacityDecision[]>();
    decisions.forEach((item) => {
      const terminal = item.terminal_code && item.terminal_code !== '<UNKNOWN>' ? item.terminal_code : null;
      const key = terminal || item.port_code || 'Zone non renseignée';
      grouped.set(key, [...(grouped.get(key) ?? []), item]);
    });
    const terminals = [...grouped.entries()]
      .sort((left, right) => right[1].length - left[1].length)
      .slice(0, 8);
    const horizons: Array<{ label: string; read: (item: CapacityDecision) => number }> = [
      { label: 'Maintenant', read: (item) => item.risk_score },
      { label: '+ 6 h', read: (item) => item.hazard_6h },
      { label: '+ 12 h', read: (item) => item.hazard_12h },
      { label: '+ 24 h', read: (item) => item.hazard_24h },
    ];
    return {
      terminals: terminals.map(([terminal]) => terminal),
      horizons: horizons.map((horizon) => horizon.label),
      values: terminals.flatMap(([, items], terminalIndex) => horizons.map((horizon, horizonIndex) => ({
        value: [
          horizonIndex,
          terminalIndex,
          Number((average(items.map(horizon.read)) * 100).toFixed(3)),
        ],
        activeCalls: Math.round(average(items.map((item) => item.active_calls))),
        capacity: Math.round(average(items.map((item) => item.capacity))),
      }))),
    };
  }, [decisions]);

  const option = useMemo(() => ({
    animationDuration: 1050,
    animationEasing: 'cubicOut',
    tooltip: {
      position: 'top',
      backgroundColor: 'rgba(8,9,12,0.96)',
      borderColor: '#1E3A5F',
      textStyle: { color: '#F8FAFC', fontFamily: 'Inter, sans-serif', fontSize: 11 },
      formatter: (params: { value: [number, number, number]; data: { activeCalls: number; capacity: number } }) => {
        const [x, y, value] = params.value;
        return `<strong>${matrix.terminals[y]}</strong><br/>${matrix.horizons[x]} : ${formatPercent(value)}<br/>${params.data.activeCalls} escales · capacité ${params.data.capacity}`;
      },
    },
    grid: { left: 108, right: 30, top: 18, bottom: 58 },
    xAxis: {
      type: 'category',
      data: matrix.horizons,
      splitArea: { show: true, areaStyle: { color: ['transparent'] } },
      axisLine: { lineStyle: { color: '#1E3A5F' } },
      axisTick: { show: false },
      axisLabel: { color: '#CBD5E1', fontFamily: 'Inter, sans-serif', fontSize: 10 },
    },
    yAxis: {
      type: 'category',
      data: matrix.terminals,
      splitArea: { show: true, areaStyle: { color: ['transparent'] } },
      axisLine: { lineStyle: { color: '#1E3A5F' } },
      axisTick: { show: false },
      axisLabel: { color: '#CBD5E1', fontFamily: 'JetBrains Mono, monospace', fontSize: 9 },
    },
    visualMap: {
      min: 0,
      max: 100,
      calculable: false,
      orient: 'horizontal',
      left: 'center',
      bottom: 3,
      text: ['Critique', 'Stable'],
      textStyle: { color: '#94A3B8', fontSize: 9 },
      inRange: { color: ['#123D5A', '#2F7FA3', '#F0C66E', '#DB6A8F'] },
    },
    series: [{
      type: 'heatmap',
      data: matrix.values,
      animationDelay: (index: number) => index * 24,
      label: {
        show: true,
        color: '#F8FAFC',
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: 10,
        formatter: (params: { value: [number, number, number] }) => formatPercent(params.value[2]),
      },
      itemStyle: { borderColor: '#08090C', borderWidth: 4, borderRadius: 6 },
      emphasis: { itemStyle: { borderColor: '#F8FAFC', borderWidth: 1 } },
    }],
  }), [matrix]);

  if (!matrix.terminals.length) return null;
  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export default CapacityHeatmap;
