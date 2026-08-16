import { useTheme } from '@mui/material';
import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from 'echarts/components';
import { LineChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import type { ForecastPoint, WeatherPoint } from 'types/replay';
import { formatShortTimestamp } from 'helpers/maritime';

echarts.use([
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DataZoomComponent,
  LineChart,
  CanvasRenderer,
]);

interface WeatherImpactChartProps {
  weather: WeatherPoint[];
  timeline?: ForecastPoint[];
  height?: number;
}

const WeatherImpactChart = ({ weather, timeline = [], height = 380 }: WeatherImpactChartProps) => {
  const theme = useTheme();

  const option = useMemo(
    () => ({
      animationDuration: 450,
      color: ['#167D85', '#D97706', '#1D4ED8'],
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
      grid: { top: 42, left: 12, right: 18, bottom: 46, containLabel: true },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: weather.map((item) => formatShortTimestamp(item.observed_at)),
        axisTick: { show: false },
        axisLabel: { color: theme.palette.text.secondary, hideOverlap: true },
        axisLine: { lineStyle: { color: '#D8E2EA' } },
      },
      yAxis: [
        {
          type: 'value',
          name: 'Mer / vent',
          min: 0,
          axisLabel: { color: theme.palette.text.secondary },
          nameTextStyle: { color: theme.palette.text.secondary },
          splitLine: { lineStyle: { color: '#E7EDF2' } },
        },
        {
          type: 'value',
          name: 'Arrivées',
          min: 0,
          axisLabel: { color: theme.palette.text.secondary },
          nameTextStyle: { color: theme.palette.text.secondary },
          splitLine: { show: false },
        },
      ],
      dataZoom: [{ type: 'inside', start: 55, end: 100 }],
      series: [
        {
          name: 'Vagues (m)',
          type: 'line',
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 2.5 },
          areaStyle: { color: 'rgba(22,125,133,0.10)' },
          data: weather.map((item) => item.wave_height_m),
        },
        {
          name: 'Vent (m/s)',
          type: 'line',
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 2 },
          data: weather.map((item) => item.wind_speed_ms),
        },
        {
          name: 'Prévision P50',
          type: 'line',
          yAxisIndex: 1,
          showSymbol: false,
          connectNulls: true,
          lineStyle: { width: 2, type: 'dashed' },
          data: weather.map((item) => {
            const matching = timeline.find(
              (point) =>
                new Date(point.as_of_time).getTime() === new Date(item.observed_at).getTime(),
            );
            return matching?.p50 ?? null;
          }),
        },
      ],
    }),
    [theme, timeline, weather],
  );

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export default WeatherImpactChart;
