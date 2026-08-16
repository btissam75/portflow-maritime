import { useTheme } from '@mui/material';
import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import {
  DataZoomComponent,
  DataZoomComponentOption,
  GridComponent,
  GridComponentOption,
  LegendComponent,
  LegendComponentOption,
  MarkLineComponent,
  MarkLineComponentOption,
  TooltipComponent,
  TooltipComponentOption,
} from 'echarts/components';
import { LineChart, LineSeriesOption } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import type { ForecastPoint } from 'types/replay';
import { formatShortTimestamp } from 'helpers/maritime';

echarts.use([
  TooltipComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  DataZoomComponent,
  LineChart,
  CanvasRenderer,
]);

type EChartsOption = echarts.ComposeOption<
  | TooltipComponentOption
  | GridComponentOption
  | LegendComponentOption
  | MarkLineComponentOption
  | DataZoomComponentOption
  | LineSeriesOption
>;

interface ForecastBandChartProps {
  data: ForecastPoint[];
  activeAsOf: string;
  height?: number;
}

const ForecastBandChart = ({ data, activeAsOf, height = 390 }: ForecastBandChartProps) => {
  const theme = useTheme();

  const option = useMemo<EChartsOption>(() => {
    const labels = data.map((point) => formatShortTimestamp(point.as_of_time));
    const lower = data.map((point) => Number(point.p10.toFixed(2)));
    const interval = data.map((point) => Number((point.p90 - point.p10).toFixed(2)));
    const activeIndex = data.findIndex((point) => point.as_of_time === activeAsOf);

    return {
      animationDuration: 850,
      animationDurationUpdate: 520,
      animationEasing: 'cubicOut',
      color: ['#0F766E', '#1D4ED8', '#D97706'],
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#102A43',
        borderWidth: 0,
        textStyle: { color: '#FFFFFF' },
        valueFormatter: (value) => `${Number(value).toFixed(1)} navires`,
      },
      legend: {
        top: 0,
        right: 0,
        itemWidth: 18,
        textStyle: { color: theme.palette.text.secondary },
        data: ['Prévision P50', 'Réalisé'],
      },
      grid: {
        top: 42,
        left: 12,
        right: 12,
        bottom: 50,
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: labels,
        axisLine: { lineStyle: { color: '#D8E2EA' } },
        axisTick: { show: false },
        axisLabel: { color: theme.palette.text.secondary, hideOverlap: true },
      },
      yAxis: {
        type: 'value',
        min: 0,
        name: 'Arrivées',
        nameTextStyle: { color: theme.palette.text.secondary },
        axisLabel: { color: theme.palette.text.secondary },
        splitLine: { lineStyle: { color: '#E7EDF2' } },
      },
      dataZoom: [
        { type: 'inside', start: 55, end: 100 },
        {
          type: 'slider',
          height: 18,
          bottom: 4,
          borderColor: 'transparent',
          backgroundColor: '#EEF3F6',
          fillerColor: 'rgba(15,118,110,0.18)',
          handleStyle: { color: '#0F766E' },
        },
      ],
      series: [
        {
          name: 'P10',
          type: 'line',
          stack: 'confidence',
          data: lower,
          symbol: 'none',
          lineStyle: { opacity: 0 },
          areaStyle: { opacity: 0 },
          silent: true,
        },
        {
          name: 'Intervalle P10-P90',
          type: 'line',
          stack: 'confidence',
          data: interval,
          symbol: 'none',
          lineStyle: { opacity: 0 },
          areaStyle: { color: 'rgba(15,118,110,0.20)' },
          silent: true,
        },
        {
          name: 'Prévision P50',
          type: 'line',
          data: data.map((point) => Number(point.p50.toFixed(2))),
          symbol: 'none',
          lineStyle: { width: 3, color: '#0F766E' },
          markLine:
            activeIndex >= 0
              ? {
                  symbol: 'none',
                  label: { show: false },
                  lineStyle: { color: '#D97706', type: 'dashed', width: 2 },
                  data: [{ xAxis: activeIndex }],
                }
              : undefined,
        },
        {
          name: 'Réalisé',
          type: 'line',
          data: data.map((point) => point.actual_arrivals),
          symbol: 'none',
          lineStyle: { width: 2, color: '#1D4ED8' },
        },
      ],
    };
  }, [activeAsOf, data, theme]);

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export default ForecastBandChart;
