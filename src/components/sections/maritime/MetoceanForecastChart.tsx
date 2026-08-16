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
import { LineChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import { formatShortTimestamp } from 'helpers/maritime';
import type { MetoceanForecastPoint } from 'types/metocean';

echarts.use([
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DataZoomComponent,
  MarkLineComponent,
  LineChart,
  CanvasRenderer,
]);

interface MetoceanForecastChartProps {
  data: MetoceanForecastPoint[];
  unit: string;
  threshold?: number;
  height?: number;
  color?: string;
}

const MetoceanForecastChart = ({
  data,
  unit,
  threshold,
  height = 360,
  color = '#0D7C76',
}: MetoceanForecastChartProps) => {
  const theme = useTheme();

  const option = useMemo(() => {
    const sorted = [...data].sort(
      (left, right) => new Date(left.valid_at).getTime() - new Date(right.valid_at).getTime(),
    );
    const lower = sorted.map((item) => item.p10);
    const interval = sorted.map((item) =>
      item.p10 != null && item.p90 != null ? item.p90 - item.p10 : null,
    );
    const hasInterval = interval.some((value) => value != null);

    return {
      animationDuration: 450,
      color: [color, color, color],
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#102A43',
        borderWidth: 0,
        textStyle: { color: '#FFFFFF' },
      },
      legend: {
        top: 0,
        right: 0,
        data: hasInterval ? ['P50', 'Intervalle P10-P90'] : ['P50'],
        textStyle: { color: theme.palette.text.secondary },
      },
      grid: { top: 42, left: 12, right: 18, bottom: 48, containLabel: true },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: sorted.map((item) => formatShortTimestamp(item.valid_at)),
        axisTick: { show: false },
        axisLabel: { color: theme.palette.text.secondary, hideOverlap: true },
        axisLine: { lineStyle: { color: '#CFDCE5' } },
      },
      yAxis: {
        type: 'value',
        name: unit,
        scale: true,
        axisLabel: { color: theme.palette.text.secondary },
        nameTextStyle: { color: theme.palette.text.secondary },
        splitLine: { lineStyle: { color: '#E7EDF2' } },
      },
      dataZoom: [{ type: 'inside', start: 0, end: 100 }],
      series: [
        ...(hasInterval
          ? [
              {
                name: 'Borne basse',
                type: 'line',
                stack: 'confidence',
                symbol: 'none',
                lineStyle: { opacity: 0 },
                areaStyle: { opacity: 0 },
                data: lower,
                tooltip: { show: false },
              },
              {
                name: 'Intervalle P10-P90',
                type: 'line',
                stack: 'confidence',
                symbol: 'none',
                lineStyle: { opacity: 0 },
                areaStyle: { color: `${color}33` },
                data: interval,
              },
            ]
          : []),
        {
          name: 'P50',
          type: 'line',
          smooth: 0.18,
          symbol: 'circle',
          symbolSize: 5,
          showSymbol: sorted.length <= 24,
          lineStyle: { width: 2.5 },
          itemStyle: { color },
          data: sorted.map((item) => item.p50),
          markLine:
            threshold == null
              ? undefined
              : {
                  silent: true,
                  symbol: 'none',
                  label: { formatter: `Seuil ${threshold} ${unit}`, color: '#A13D2D' },
                  lineStyle: { color: '#D0604C', type: 'dashed', width: 1.5 },
                  data: [{ yAxis: threshold }],
                },
        },
      ],
    };
  }, [color, data, threshold, theme, unit]);

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export default MetoceanForecastChart;
