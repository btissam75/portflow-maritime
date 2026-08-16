import { useTheme } from '@mui/material';
import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import {
  GridComponent,
  GridComponentOption,
  LegendComponent,
  LegendComponentOption,
  TooltipComponent,
  TooltipComponentOption,
} from 'echarts/components';
import { BarChart, BarSeriesOption, LineChart, LineSeriesOption } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import type { ForecastPoint, HorizonMetric } from 'types/replay';

echarts.use([
  TooltipComponent,
  GridComponent,
  LegendComponent,
  BarChart,
  LineChart,
  CanvasRenderer,
]);

type EChartsOption = echarts.ComposeOption<
  | TooltipComponentOption
  | GridComponentOption
  | LegendComponentOption
  | BarSeriesOption
  | LineSeriesOption
>;

interface HorizonComparisonChartProps {
  forecasts: ForecastPoint[];
  metrics: HorizonMetric[];
  height?: number;
}

const HorizonComparisonChart = ({
  forecasts,
  metrics,
  height = 250,
}: HorizonComparisonChartProps) => {
  const theme = useTheme();

  const option = useMemo<EChartsOption>(() => {
    const horizons = [6, 12, 24];
    const findForecast = (horizon: number) =>
      forecasts.find((item) => item.horizon_h === horizon);
    const findMetric = (horizon: number) => metrics.find((item) => item.horizon_h === horizon);

    return {
      animationDuration: 820,
      animationEasing: 'cubicOut',
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#102A43',
        borderWidth: 0,
        textStyle: { color: '#FFFFFF' },
      },
      legend: {
        top: 0,
        right: 0,
        itemWidth: 12,
        itemHeight: 8,
        textStyle: { color: theme.palette.text.secondary, fontSize: 10 },
      },
      grid: { top: 44, left: 8, right: 8, bottom: 6, containLabel: true },
      xAxis: {
        type: 'category',
        data: horizons.map((value) => `${value} h`),
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#D8E2EA' } },
        axisLabel: { color: theme.palette.text.secondary },
      },
      yAxis: [
        {
          type: 'value',
          min: 0,
          splitLine: { lineStyle: { color: '#EEF2F6' } },
          axisLabel: { color: theme.palette.text.secondary },
        },
        {
          type: 'value',
          min: 0,
          max: 100,
          splitLine: { show: false },
          axisLabel: { formatter: '{value} %', color: theme.palette.text.secondary },
        },
      ],
      series: [
        {
          name: 'P50',
          type: 'bar',
          data: horizons.map((value) => findForecast(value)?.p50 ?? null),
          barWidth: 18,
          itemStyle: { color: '#5D5FEF', borderRadius: [4, 4, 0, 0] },
        },
        {
          name: 'Réalisé',
          type: 'bar',
          data: horizons.map((value) => findForecast(value)?.actual_arrivals ?? null),
          barWidth: 18,
          itemStyle: { color: '#00AB9A', borderRadius: [4, 4, 0, 0] },
        },
        {
          name: 'Couverture',
          type: 'line',
          yAxisIndex: 1,
          data: horizons.map((value) => {
            const coverage = findMetric(value)?.coverage_p10_p90;
            return coverage == null ? null : Number((coverage * 100).toFixed(1));
          }),
          symbol: 'circle',
          symbolSize: 7,
          lineStyle: { color: '#FFA412', width: 2.5 },
          itemStyle: { color: '#FFA412' },
        },
      ],
    };
  }, [forecasts, metrics, theme]);

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export default HorizonComparisonChart;
