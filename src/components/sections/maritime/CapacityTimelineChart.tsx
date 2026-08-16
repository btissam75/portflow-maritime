import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components';
import { LineChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import type { CapacityDecision } from 'types/capacity';
import { portflowPalette as pf } from 'theme/portflowPalette';

echarts.use([TooltipComponent, LegendComponent, GridComponent, LineChart, CanvasRenderer]);

const CapacityTimelineChart = ({ data, height = 250 }: { data: CapacityDecision[]; height?: number }) => {
  const option = useMemo(() => {
    const riskValues = data.map((item) => Number((item.risk_score * 100).toFixed(3)));
    const riskMaximum = Math.max(1, Math.ceil(Math.max(...riskValues, 0) * 1.25 * 10) / 10);
    const showSymbols = data.length <= 8;
    return ({
    animationDuration: 1100,
    animationEasing: 'cubicOut',
    tooltip: {
      trigger: 'axis',
      backgroundColor: pf.background.navigation,
      borderColor: pf.map.road,
      borderWidth: 1,
      padding: 10,
      textStyle: {
        color: pf.text.primary,
        fontFamily: 'Inter, sans-serif',
        fontSize: 10,
      },
      axisPointer: { type: 'line', lineStyle: { color: pf.functional.cyan, type: 'dashed' } },
    },
    legend: {
      top: 4,
      right: 4,
      itemWidth: 16,
      itemHeight: 6,
      textStyle: { color: pf.text.secondary, fontFamily: 'Inter, sans-serif', fontSize: 10 },
      data: ['Risque', 'Temps P50', 'Incertitude P10-P90'],
    },
    grid: { left: 46, right: 46, top: 42, bottom: 28 },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: data.map((item) => new Intl.DateTimeFormat('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        hour: '2-digit',
        timeZone: 'Africa/Casablanca',
      }).format(new Date(item.decision_at))),
      axisLine: { lineStyle: { color: pf.map.outline } },
      axisLabel: { color: pf.text.secondary, fontSize: 10, hideOverlap: true },
    },
    yAxis: [
      {
        type: 'value', min: 0, max: riskMaximum, name: 'Risque %',
        nameTextStyle: { color: pf.text.tertiary, fontSize: 10 },
        axisLabel: { color: pf.text.secondary, fontSize: 10, formatter: (value: number) => `${value.toLocaleString('fr-FR', { maximumFractionDigits: 2 })}%` },
        splitLine: { lineStyle: { color: pf.structure.divider } },
      },
      {
        type: 'value', name: 'Heures',
        nameTextStyle: { color: pf.text.tertiary, fontSize: 10 },
        axisLabel: { color: pf.text.secondary, fontSize: 10 },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: 'Risque', type: 'line', smooth: 0.32, showSymbol: showSymbols, symbolSize: 7,
        data: riskValues,
        lineStyle: { color: pf.functional.red, width: 2.2 },
        itemStyle: { color: pf.functional.red },
      },
      {
        name: 'Base P10', type: 'line', yAxisIndex: 1, stack: 'confidence',
        smooth: 0.28, showSymbol: false, silent: true,
        data: data.map((item) => Number(item.remaining_p10_h.toFixed(1))),
        lineStyle: { opacity: 0 },
        areaStyle: { opacity: 0 },
        tooltip: { show: false },
      },
      {
        name: 'Incertitude P10-P90', type: 'line', yAxisIndex: 1, stack: 'confidence',
        smooth: 0.28, showSymbol: false, silent: true,
        data: data.map((item) => Number(Math.max(0, item.remaining_p90_h - item.remaining_p10_h).toFixed(1))),
        lineStyle: { opacity: 0 },
        areaStyle: { color: 'rgba(73,167,255,0.22)' },
        tooltip: { show: false },
      },
      {
        name: 'P10', type: 'line', yAxisIndex: 1, smooth: 0.28, showSymbol: showSymbols, symbolSize: 5,
        data: data.map((item) => Number(item.remaining_p10_h.toFixed(1))),
        lineStyle: { color: pf.functional.blue, width: 1, type: 'dashed' },
        itemStyle: { color: pf.functional.blue },
      },
      {
        name: 'P90', type: 'line', yAxisIndex: 1, smooth: 0.28, showSymbol: showSymbols, symbolSize: 5,
        data: data.map((item) => Number(item.remaining_p90_h.toFixed(1))),
        lineStyle: { color: pf.functional.amber, width: 1, type: 'dashed' },
        itemStyle: { color: pf.functional.amber },
      },
      {
        name: 'Temps P50', type: 'line', yAxisIndex: 1, smooth: 0.3, showSymbol: showSymbols, symbolSize: 7,
        data: data.map((item) => Number(item.remaining_p50_h.toFixed(1))),
        lineStyle: { color: pf.functional.cyan, width: 2.3 },
        itemStyle: { color: pf.functional.cyan },
      },
    ],
  });
  }, [data]);

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export default CapacityTimelineChart;
