import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components';
import { LineChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import type { CapacityDecision } from 'types/capacity';
import { portflowPalette as pf } from 'theme/portflowPalette';

echarts.use([TooltipComponent, LegendComponent, GridComponent, LineChart, CanvasRenderer]);

interface CapacityEtaBandChartProps {
  data: CapacityDecision[];
  fallback?: CapacityDecision;
  height?: number;
}

const formatTime = (value: string) => new Intl.DateTimeFormat('fr-FR', {
  day: '2-digit',
  month: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  timeZone: 'Africa/Casablanca',
}).format(new Date(value));

const CapacityEtaBandChart = ({ data, fallback, height = 210 }: CapacityEtaBandChartProps) => {
  const points = useMemo(() => {
    const source = data.length ? [...data] : fallback ? [fallback] : [];
    return source
      .sort((left, right) => new Date(left.decision_at).getTime() - new Date(right.decision_at).getTime())
      .slice(-36);
  }, [data, fallback]);

  const option = useMemo(() => ({
    animationDuration: 650,
    tooltip: {
      trigger: 'axis',
      backgroundColor: pf.background.navigation,
      borderColor: pf.map.road,
      borderWidth: 1,
      textStyle: { color: pf.text.primary, fontFamily: 'Inter, sans-serif', fontSize: 11 },
      valueFormatter: (value: number) => `${Number(value).toFixed(1)} h`,
    },
    legend: {
      top: 0,
      right: 0,
      itemWidth: 15,
      itemHeight: 6,
      data: ['P10', 'P50', 'P90', 'Intervalle P10-P90'],
      textStyle: { color: pf.text.secondary, fontFamily: 'Inter, sans-serif', fontSize: 10 },
    },
    grid: { left: 38, right: 12, top: 38, bottom: 30 },
    xAxis: {
      type: 'category',
      boundaryGap: points.length === 1,
      data: points.map((item) => formatTime(item.decision_at)),
      axisLine: { lineStyle: { color: pf.map.outline } },
      axisTick: { show: false },
      axisLabel: { color: pf.text.secondary, fontSize: 10, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      name: 'heures',
      min: 0,
      nameTextStyle: { color: pf.text.tertiary, fontSize: 10 },
      axisLabel: { color: pf.text.secondary, fontSize: 10 },
      splitLine: { lineStyle: { color: pf.structure.divider } },
    },
    series: [
      {
        name: 'Base intervalle',
        type: 'line',
        stack: 'eta-confidence',
        silent: true,
        showSymbol: false,
        data: points.map((item) => Number(item.remaining_p10_h.toFixed(1))),
        lineStyle: { opacity: 0 },
        areaStyle: { opacity: 0 },
        tooltip: { show: false },
      },
      {
        name: 'Intervalle P10-P90',
        type: 'line',
        stack: 'eta-confidence',
        silent: true,
        showSymbol: false,
        data: points.map((item) => Number(Math.max(0, item.remaining_p90_h - item.remaining_p10_h).toFixed(1))),
        lineStyle: { opacity: 0 },
        areaStyle: { color: 'rgba(73,167,255,0.22)' },
        tooltip: { show: false },
      },
      {
        name: 'P10',
        type: 'line',
        smooth: 0.24,
        showSymbol: points.length === 1,
        symbolSize: 7,
        data: points.map((item) => Number(item.remaining_p10_h.toFixed(1))),
        lineStyle: { color: pf.functional.blue, width: 1.3, type: 'dashed' },
        itemStyle: { color: pf.functional.blue },
      },
      {
        name: 'P90',
        type: 'line',
        smooth: 0.24,
        showSymbol: points.length === 1,
        symbolSize: 7,
        data: points.map((item) => Number(item.remaining_p90_h.toFixed(1))),
        lineStyle: { color: pf.functional.amber, width: 1.3, type: 'dashed' },
        itemStyle: { color: pf.functional.amber },
      },
      {
        name: 'P50',
        type: 'line',
        smooth: 0.28,
        showSymbol: points.length === 1,
        symbolSize: 8,
        data: points.map((item) => Number(item.remaining_p50_h.toFixed(1))),
        lineStyle: { color: pf.functional.cyan, width: 2.6 },
        itemStyle: { color: pf.functional.cyan },
      },
    ],
  }), [points]);

  if (!points.length) return null;
  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export default CapacityEtaBandChart;
