import { useTheme } from '@mui/material';
import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components';
import { BarChart, LineChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import type { HorizonMetric } from 'types/replay';

echarts.use([TooltipComponent, GridComponent, LegendComponent, BarChart, LineChart, CanvasRenderer]);

interface ModelPerformanceChartProps {
  metrics: HorizonMetric[];
  height?: number;
}

const ModelPerformanceChart = ({ metrics, height = 350 }: ModelPerformanceChartProps) => {
  const theme = useTheme();
  const option = useMemo(
    () => ({
      animationDuration: 500,
      color: ['#0F766E', '#D97706', '#1D4ED8'],
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#102A43',
        borderWidth: 0,
        textStyle: { color: '#FFFFFF' },
      },
      legend: {
        top: 0,
        right: 0,
        textStyle: { color: theme.palette.text.secondary },
      },
      grid: { top: 42, left: 16, right: 16, bottom: 18, containLabel: true },
      xAxis: {
        type: 'category',
        data: metrics.map((item) => `${item.horizon_h} h`),
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#D8E2EA' } },
        axisLabel: { color: theme.palette.text.secondary },
      },
      yAxis: [
        {
          type: 'value',
          name: 'Erreur',
          min: 0,
          splitLine: { lineStyle: { color: '#E7EDF2' } },
          axisLabel: { color: theme.palette.text.secondary },
          nameTextStyle: { color: theme.palette.text.secondary },
        },
        {
          type: 'value',
          name: 'Couverture',
          min: 0,
          max: 100,
          axisLabel: { formatter: '{value} %', color: theme.palette.text.secondary },
          nameTextStyle: { color: theme.palette.text.secondary },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: 'MAE',
          type: 'bar',
          barMaxWidth: 32,
          data: metrics.map((item) => Number(item.mae.toFixed(2))),
          itemStyle: { borderRadius: [4, 4, 0, 0] },
        },
        {
          name: 'RMSE',
          type: 'bar',
          barMaxWidth: 32,
          data: metrics.map((item) => Number(item.rmse.toFixed(2))),
          itemStyle: { borderRadius: [4, 4, 0, 0] },
        },
        {
          name: 'Couverture P10-P90',
          type: 'line',
          yAxisIndex: 1,
          symbolSize: 9,
          data: metrics.map((item) => Number((item.coverage_p10_p90 * 100).toFixed(1))),
          lineStyle: { width: 3 },
        },
      ],
    }),
    [metrics, theme],
  );

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export default ModelPerformanceChart;
