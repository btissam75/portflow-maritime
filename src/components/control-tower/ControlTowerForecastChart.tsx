import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components';
import { LineChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import useReducedMotion from 'hooks/useReducedMotion';
import { portflowPalette as pf } from 'theme/portflowPalette';

echarts.use([TooltipComponent, LegendComponent, GridComponent, LineChart, CanvasRenderer]);

const labels = ['-6 h', '-4 h', '-2 h', 'Maintenant', '+2 h', '+4 h', '+6 h', '+8 h', '+10 h', '+12 h'];
const realized = [21, 23, 25, 27, null, null, null, null, null, null];
const p50 = [null, null, null, 27, 29, 31, 34, 36, 38, 41];
const p10 = [null, null, null, 27, 26, 28, 30, 31, 33, 35];
const p90 = [null, null, null, 27, 33, 36, 39, 43, 46, 51];

const ControlTowerForecastChart = ({ height = 220 }: { height?: number }) => {
  const reducedMotion = useReducedMotion();
  const option = useMemo(() => ({
    animation: !reducedMotion,
    animationDuration: 620,
    animationEasing: 'cubicOut',
    tooltip: {
      trigger: 'axis',
      backgroundColor: pf.background.navigation,
      borderColor: pf.map.road,
      borderWidth: 1,
      padding: 10,
      textStyle: { color: pf.text.primary, fontFamily: 'Inter, sans-serif', fontSize: 12 },
      axisPointer: { type: 'line', lineStyle: { color: pf.functional.cyan, opacity: 0.45 } },
    },
    legend: {
      top: 2,
      right: 6,
      itemWidth: 18,
      itemHeight: 4,
      textStyle: { color: pf.text.secondary, fontSize: 11 },
      data: ['Réalisé', 'P50', 'P10–P90'],
    },
    grid: { left: 42, right: 20, top: 36, bottom: 28 },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: labels,
      axisLine: { lineStyle: { color: pf.map.outline } },
      axisTick: { show: false },
      axisLabel: { color: pf.text.secondary, fontSize: 11, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      name: 'unités / h',
      min: 15,
      nameTextStyle: { color: pf.text.tertiary, fontSize: 11 },
      axisLabel: { color: pf.text.secondary, fontSize: 11 },
      splitLine: { lineStyle: { color: pf.structure.divider } },
    },
    series: [
      {
        name: 'Réalisé', type: 'line', data: realized, smooth: 0.28, showSymbol: false,
        lineStyle: { color: pf.text.primary, width: 2 }, itemStyle: { color: pf.text.primary },
      },
      {
        name: 'Base P10', type: 'line', stack: 'band', data: p10, showSymbol: false, silent: true,
        lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 }, tooltip: { show: false },
      },
      {
        name: 'P10–P90', type: 'line', stack: 'band', showSymbol: false, silent: true,
        data: p90.map((value, index) => value == null || p10[index] == null ? null : value - (p10[index] as number)),
        lineStyle: { opacity: 0 }, areaStyle: { color: 'rgba(73,167,255,0.22)' }, tooltip: { show: false },
      },
      {
        name: 'P50', type: 'line', data: p50, smooth: 0.3, showSymbol: false,
        lineStyle: { color: pf.functional.cyan, width: 3 }, itemStyle: { color: pf.functional.cyan },
      },
      {
        name: 'Seuil opérationnel', type: 'line', data: labels.map(() => 45), showSymbol: false, silent: true,
        lineStyle: { color: pf.functional.amber, width: 1.5, type: 'dashed' }, tooltip: { show: false },
      },
    ],
  }), [reducedMotion]);

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export default ControlTowerForecastChart;
