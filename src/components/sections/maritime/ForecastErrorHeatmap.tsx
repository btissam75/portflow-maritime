import { useTheme } from '@mui/material';
import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import {
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components';
import { HeatmapChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import type { ErrorHeatmapCell } from 'types/replay';

echarts.use([
  TooltipComponent,
  GridComponent,
  VisualMapComponent,
  HeatmapChart,
  CanvasRenderer,
]);

interface ForecastErrorHeatmapProps {
  data: ErrorHeatmapCell[];
  height?: number;
}

interface HeatmapTooltipParam {
  data?: [number, number, number, number, number, number];
}

const days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'];
const hours = Array.from({ length: 24 }, (_value, index) => `${index}h`);

const ForecastErrorHeatmap = ({
  data,
  height = 330,
}: ForecastErrorHeatmapProps) => {
  const theme = useTheme();

  const option = useMemo(() => {
    const values: Array<[number, number, number, number, number, number]> = data.map(
      (cell) => [
        cell.hour_of_day,
        cell.day_of_week - 1,
        Number(cell.mae.toFixed(2)),
        cell.observations,
        Number(cell.bias.toFixed(2)),
        Number((cell.coverage_p10_p90 * 100).toFixed(1)),
      ],
    );
    const maxMae = Math.max(1, ...data.map((cell) => cell.mae));

    return {
      animationDuration: 760,
      animationEasing: 'cubicOut',
      tooltip: {
        position: 'top',
        backgroundColor: '#102A43',
        borderWidth: 0,
        textStyle: { color: '#FFFFFF' },
        formatter: (raw: unknown) => {
          const params = raw as HeatmapTooltipParam;
          const point = params.data;
          if (!point) return '';
          return [
            `<strong>${days[point[1]]} ${hours[point[0]]}</strong>`,
            `MAE : ${point[2].toFixed(2)} navires`,
            `Biais : ${point[4].toFixed(2)}`,
            `Couverture : ${point[5].toFixed(1)} %`,
            `${point[3]} observations`,
          ].join('<br/>');
        },
      },
      grid: { top: 18, left: 18, right: 18, bottom: 64, containLabel: true },
      xAxis: {
        type: 'category',
        data: hours,
        splitArea: { show: true },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#D8E2EA' } },
        axisLabel: {
          color: theme.palette.text.secondary,
          interval: 2,
        },
      },
      yAxis: {
        type: 'category',
        data: days,
        splitArea: { show: true },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#D8E2EA' } },
        axisLabel: { color: theme.palette.text.secondary },
      },
      visualMap: {
        min: 0,
        max: Number(maxMae.toFixed(1)),
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 4,
        itemWidth: 14,
        itemHeight: 150,
        text: ['Erreur forte', 'Faible'],
        textStyle: { color: theme.palette.text.secondary, fontSize: 10 },
        inRange: {
          color: ['#DDF4F1', '#7CC9BE', '#F5C36A', '#E5484D'],
        },
      },
      series: [
        {
          name: 'MAE',
          type: 'heatmap',
          data: values,
          label: { show: false },
          itemStyle: {
            borderColor: '#FFFFFF',
            borderWidth: 2,
            borderRadius: 2,
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(16,42,67,0.25)',
            },
          },
        },
      ],
    };
  }, [data, theme]);

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export default ForecastErrorHeatmap;
