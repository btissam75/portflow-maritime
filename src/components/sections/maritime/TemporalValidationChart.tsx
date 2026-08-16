import { useTheme } from '@mui/material';
import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
} from 'echarts/components';
import { BarChart, LineChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import { formatShortTimestamp } from 'helpers/maritime';
import type { PerformancePoint } from 'types/replay';

echarts.use([
  TooltipComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  DataZoomComponent,
  BarChart,
  LineChart,
  CanvasRenderer,
]);

interface TemporalValidationChartProps {
  data: PerformancePoint[];
  height?: number;
}

const TemporalValidationChart = ({
  data,
  height = 330,
}: TemporalValidationChartProps) => {
  const theme = useTheme();

  const option = useMemo(
    () => ({
      animationDuration: 900,
      animationDurationUpdate: 520,
      animationEasing: 'cubicOut',
      color: ['#1D4ED8', '#0F766E', '#D97706'],
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#102A43',
        borderWidth: 0,
        textStyle: { color: '#FFFFFF' },
      },
      legend: {
        top: 0,
        right: 0,
        itemWidth: 14,
        textStyle: { color: theme.palette.text.secondary, fontSize: 10 },
      },
      grid: { top: 44, left: 12, right: 16, bottom: 46, containLabel: true },
      xAxis: {
        type: 'category',
        boundaryGap: true,
        data: data.map((point) => formatShortTimestamp(point.period_start)),
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#D8E2EA' } },
        axisLabel: { color: theme.palette.text.secondary, hideOverlap: true },
      },
      yAxis: [
        {
          type: 'value',
          name: 'MAE',
          min: 0,
          nameTextStyle: { color: theme.palette.text.secondary },
          axisLabel: { color: theme.palette.text.secondary },
          splitLine: { lineStyle: { color: '#E7EDF2' } },
        },
        {
          type: 'value',
          name: 'Couverture',
          min: 0,
          max: 100,
          nameTextStyle: { color: theme.palette.text.secondary },
          axisLabel: {
            color: theme.palette.text.secondary,
            formatter: '{value} %',
          },
          splitLine: { show: false },
        },
      ],
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        {
          type: 'slider',
          height: 16,
          bottom: 2,
          borderColor: 'transparent',
          backgroundColor: '#EEF3F6',
          fillerColor: 'rgba(29,78,216,0.14)',
          handleStyle: { color: '#1D4ED8' },
        },
      ],
      series: [
        {
          name: 'MAE quotidienne',
          type: 'bar',
          barMaxWidth: 11,
          data: data.map((point) => Number(point.mae.toFixed(2))),
          itemStyle: {
            color: '#A9C5FF',
            borderRadius: [3, 3, 0, 0],
          },
        },
        {
          name: 'Tendance MAE',
          type: 'line',
          smooth: 0.35,
          showSymbol: false,
          data: data.map((point) => Number(point.mae.toFixed(2))),
          lineStyle: { width: 2.5, color: '#1D4ED8' },
        },
        {
          name: 'Couverture P10-P90',
          type: 'line',
          yAxisIndex: 1,
          smooth: 0.3,
          showSymbol: false,
          data: data.map((point) =>
            Number((point.coverage_p10_p90 * 100).toFixed(1)),
          ),
          lineStyle: { width: 2.5, color: '#0F766E' },
          areaStyle: { color: 'rgba(15,118,110,0.08)' },
          markLine: {
            silent: true,
            symbol: 'none',
            label: {
              formatter: 'Cible 80 %',
              color: '#D97706',
              fontSize: 10,
            },
            lineStyle: { color: '#D97706', type: 'dashed', width: 1.5 },
            data: [{ yAxis: 80 }],
          },
        },
      ],
    }),
    [data, theme],
  );

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export default TemporalValidationChart;
