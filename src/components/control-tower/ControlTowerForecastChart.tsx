import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from 'echarts/components';
import { BarChart, LineChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import type { TowerForecastPoint } from 'types/controlTower';
import { portflowPalette as pf } from 'theme/portflowPalette';

echarts.use([
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  BarChart,
  LineChart,
  CanvasRenderer,
]);

const ControlTowerForecastChart = ({
  data,
  height = 390,
}: {
  data: TowerForecastPoint[];
  height?: number;
}) => {
  const option = useMemo(() => {
    const labels = data.map((point) => `H+${point.horizon_h}`);
    const band = data.map((point) => point.backlog_p90 - point.backlog_p10);
    return {
      animationDuration: 1200,
      animationEasing: 'cubicOut',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(13,15,18,.97)',
        borderColor: 'rgba(85,214,194,.35)',
        borderWidth: 1,
        padding: [12, 14],
        extraCssText: 'border-radius:12px;box-shadow:0 18px 45px rgba(0,0,0,.4)',
        textStyle: { color: pf.text.primary },
      },
      legend: {
        top: 0,
        left: 0,
        itemWidth: 18,
        itemHeight: 4,
        textStyle: { color: pf.text.secondary, fontSize: 10 },
      },
      grid: { top: 50, left: 12, right: 14, bottom: 54, containLabel: true },
      xAxis: {
        type: 'category',
        data: labels,
        axisTick: { show: false },
        axisLine: { lineStyle: { color: pf.structure.border } },
        axisLabel: { color: pf.text.tertiary, fontSize: 9 },
      },
      yAxis: [
        {
          type: 'value',
          name: 'Backlog / capacité',
          nameTextStyle: { color: pf.text.tertiary, fontSize: 9 },
          axisLabel: { color: pf.text.tertiary, fontSize: 9 },
          splitLine: { lineStyle: { color: pf.structure.divider } },
        },
        {
          type: 'value',
          name: 'Flux horaire',
          nameTextStyle: { color: pf.text.tertiary, fontSize: 9 },
          axisLabel: { color: pf.text.tertiary, fontSize: 9 },
          splitLine: { show: false },
        },
      ],
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        {
          type: 'slider',
          start: 0,
          end: 100,
          height: 16,
          bottom: 4,
          borderColor: pf.structure.border,
          backgroundColor: 'rgba(8,9,11,.55)',
          fillerColor: 'rgba(85,214,194,.15)',
          handleStyle: { color: pf.functional.cyan },
          textStyle: { color: pf.text.tertiary, fontSize: 8 },
        },
      ],
      series: [
        {
          name: 'Borne basse',
          type: 'line',
          data: data.map((point) => point.backlog_p10),
          stack: 'interval',
          symbol: 'none',
          lineStyle: { opacity: 0 },
          areaStyle: { opacity: 0 },
          tooltip: { show: false },
        },
        {
          name: 'Incertitude',
          type: 'line',
          data: band,
          stack: 'interval',
          symbol: 'none',
          lineStyle: { opacity: 0 },
          areaStyle: { color: 'rgba(165,139,250,.18)' },
        },
        {
          name: 'Backlog prévu',
          type: 'line',
          data: data.map((point) => point.backlog_p50),
          smooth: 0.34,
          symbol: 'circle',
          symbolSize: 5,
          lineStyle: { color: pf.functional.purple, width: 3.4 },
          itemStyle: { color: pf.functional.purple },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(165,139,250,.22)' },
              { offset: 1, color: 'rgba(165,139,250,.01)' },
            ]),
          },
        },
        {
          name: 'Capacité normale',
          type: 'line',
          data: data.map((point) => point.normal_capacity),
          symbol: 'none',
          lineStyle: { color: pf.functional.amber, width: 2, type: 'dashed' },
        },
        {
          name: 'Capacité renforcée',
          type: 'line',
          data: data.map((point) => point.reinforced_capacity),
          symbol: 'none',
          lineStyle: { color: pf.functional.green, width: 1.8, type: 'dotted' },
        },
        {
          name: 'Arrivées',
          type: 'bar',
          yAxisIndex: 1,
          data: data.map((point) => point.arrivals),
          barMaxWidth: 8,
          itemStyle: { color: 'rgba(112,166,232,.55)', borderRadius: [4, 4, 0, 0] },
        },
        {
          name: 'Sorties',
          type: 'bar',
          yAxisIndex: 1,
          data: data.map((point) => point.departures),
          barMaxWidth: 8,
          itemStyle: { color: 'rgba(85,214,194,.52)', borderRadius: [4, 4, 0, 0] },
        },
      ],
    };
  }, [data]);

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export default ControlTowerForecastChart;
