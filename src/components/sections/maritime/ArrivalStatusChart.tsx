import { Box, Typography, useTheme } from '@mui/material';
import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import {
  LegendComponent,
  LegendComponentOption,
  TooltipComponent,
  TooltipComponentOption,
} from 'echarts/components';
import { PieChart, PieSeriesOption } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import type { PortCallItem, PortCallStatus } from 'types/replay';
import { statusLabels } from 'helpers/maritime';

echarts.use([TooltipComponent, LegendComponent, PieChart, CanvasRenderer]);

type EChartsOption = echarts.ComposeOption<
  TooltipComponentOption | LegendComponentOption | PieSeriesOption
>;

interface ArrivalStatusChartProps {
  data: PortCallItem[];
  height?: number;
}

const statusOrder: PortCallStatus[] = ['EXPECTED', 'OVERDUE', 'ARRIVED', 'BERTHED', 'DEPARTED'];
const statusPalette: Record<PortCallStatus, string> = {
  EXPECTED: '#5D5FEF',
  OVERDUE: '#FA5A7D',
  ARRIVED: '#00AB9A',
  BERTHED: '#FFA412',
  DEPARTED: '#96A5B8',
};

const ArrivalStatusChart = ({ data, height = 246 }: ArrivalStatusChartProps) => {
  const theme = useTheme();
  const activeCalls = data.filter((item) => item.status !== 'DEPARTED').length;

  const option = useMemo<EChartsOption>(() => {
    const counts = statusOrder.map((status) => ({
      name: statusLabels[status],
      value: data.filter((item) => item.status === status).length,
      itemStyle: { color: statusPalette[status] },
    }));

    return {
      animationDuration: 850,
      animationEasing: 'cubicOut',
      tooltip: {
        trigger: 'item',
        backgroundColor: '#102A43',
        borderWidth: 0,
        textStyle: { color: '#FFFFFF' },
        formatter: '{b}<br/><strong>{c}</strong> escales ({d} %)',
      },
      legend: {
        bottom: 0,
        left: 'center',
        itemWidth: 9,
        itemHeight: 9,
        itemGap: 12,
        textStyle: { color: theme.palette.text.secondary, fontSize: 10 },
      },
      series: [
        {
          name: 'Escales',
          type: 'pie',
          radius: ['57%', '78%'],
          center: ['50%', '43%'],
          avoidLabelOverlap: true,
          padAngle: 2,
          itemStyle: {
            borderColor: '#FFFFFF',
            borderWidth: 2,
            borderRadius: 4,
          },
          label: { show: false },
          emphasis: {
            scaleSize: 5,
            itemStyle: { shadowBlur: 14, shadowColor: 'rgba(16,42,67,0.18)' },
          },
          data: counts,
        },
      ],
    };
  }, [data, theme]);

  return (
    <Box position="relative">
      <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />
      <Box
        sx={{
          position: 'absolute',
          left: '50%',
          top: '43%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
          pointerEvents: 'none',
        }}
      >
        <Typography variant="h3">{activeCalls}</Typography>
        <Typography variant="caption" color="text.secondary">
          actives
        </Typography>
      </Box>
    </Box>
  );
};

export default ArrivalStatusChart;
