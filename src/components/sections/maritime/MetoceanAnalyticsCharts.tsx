import { useMemo } from 'react';
import * as echarts from 'echarts/core';
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
} from 'echarts/components';
import { BarChart, LineChart, PieChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import ReactEchart from 'components/base/ReactEhart';
import type { LiveAtmosphereHourly, LiveMarineHourly } from 'types/liveMetocean';
import type { MetoceanForecastPoint } from 'types/metocean';
import { portflowPalette as pf } from 'theme/portflowPalette';

echarts.use([
  TooltipComponent,
  GridComponent,
  LegendComponent,
  TitleComponent,
  DataZoomComponent,
  LineChart,
  BarChart,
  PieChart,
  CanvasRenderer,
]);

export type ForecastMode = 'NEXT_24H' | 'OUTLOOK_72H';
export type ProjectionFocus = 'COMBINED' | 'TEMPERATURE' | 'WAVES';

const chartText = pf.text.secondary;
const chartGrid = pf.structure.divider;

const formatHour = (timestamp: string) =>
  new Date(timestamp).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });

const groupForecast = (points: MetoceanForecastPoint[], variable: string) =>
  points
    .filter((point) => point.variable === variable)
    .sort((left, right) => left.horizon_h - right.horizon_h);

const baseOption = (labels: string[]) => ({
  animationDuration: 1350,
  animationDurationUpdate: 700,
  animationEasing: 'cubicOut',
  animationEasingUpdate: 'cubicInOut',
  tooltip: {
    trigger: 'axis',
    padding: [12, 14],
    backgroundColor: 'rgba(5,20,31,.97)',
    borderColor: 'rgba(110,214,229,.34)',
    borderWidth: 1,
    extraCssText: 'border-radius:12px;box-shadow:0 18px 48px rgba(0,0,0,.4);',
    textStyle: { color: pf.text.primary, fontFamily: 'Inter, Segoe UI, sans-serif' },
    axisPointer: {
      type: 'line',
      snap: true,
      lineStyle: { color: pf.functional.cyan, opacity: 0.7, type: 'dashed' },
      label: { show: true, color: '#06121C', backgroundColor: pf.functional.cyan },
    },
  },
  legend: {
    top: 0,
    left: 0,
    itemWidth: 20,
    itemHeight: 4,
    itemGap: 16,
    icon: 'roundRect',
    textStyle: { color: chartText, fontSize: 10.5 },
    inactiveColor: pf.text.disabled,
  },
  grid: { top: 48, left: 12, right: 14, bottom: 58, containLabel: true },
  xAxis: {
    type: 'category' as const,
    boundaryGap: false,
    data: labels,
    axisTick: { show: false },
    axisLabel: { color: chartText, hideOverlap: true, fontSize: 9.5, margin: 12 },
    axisLine: { lineStyle: { color: pf.map.outline } },
  },
  dataZoom: [
    { type: 'inside', start: 0, end: 100, zoomOnMouseWheel: true, moveOnMouseMove: true },
    {
      type: 'slider',
      start: 0,
      end: 100,
      bottom: 4,
      height: 17,
      borderColor: 'rgba(137,167,180,.18)',
      backgroundColor: 'rgba(6,18,28,.55)',
      fillerColor: 'rgba(54,214,207,.18)',
      handleStyle: { color: pf.functional.cyan, borderColor: pf.functional.cyan },
      moveHandleStyle: { color: pf.functional.cyan },
      textStyle: { color: pf.text.tertiary, fontSize: 8 },
    },
  ],
});

const axis = (name: string, side: 'left' | 'right' = 'left') => ({
  type: 'value' as const,
  name,
  position: side,
  scale: true,
  nameTextStyle: { color: chartText, fontSize: 10 },
  axisLabel: { color: chartText, fontSize: 10 },
  axisLine: { show: false },
  axisTick: { show: false },
  splitLine: { show: side === 'left', lineStyle: { color: chartGrid } },
});

const line = (
  name: string,
  data: Array<number | null>,
  color: string,
  yAxisIndex = 0,
  area = false,
) => ({
  name,
  type: 'line' as const,
  yAxisIndex,
  smooth: 0.32,
  symbol: 'circle',
  symbolSize: 5,
  showSymbol: data.length <= 8,
  connectNulls: true,
  lineStyle: { color, width: 2.8, cap: 'round', join: 'round' },
  itemStyle: { color, borderColor: pf.background.primary, borderWidth: 2 },
  areaStyle: area
    ? {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: `${color}66` },
          { offset: 1, color: `${color}05` },
        ]),
      }
    : undefined,
  emphasis: { focus: 'series' },
  blur: { lineStyle: { opacity: 0.2 }, areaStyle: { opacity: 0.04 } },
  animationDuration: 1100,
  animationDelay: (index: number) => Math.min(index * 12, 360),
  data,
});

const convertTemperature = (value: number | null, unit: 'C' | 'F') =>
  value == null ? null : unit === 'F' ? (value * 9) / 5 + 32 : value;

export const MetoceanProjectionChart = ({
  liveAtmosphere,
  liveMarine,
  horizonHours,
  focus,
  temperatureUnit,
  waveThreshold,
  height = 390,
}: {
  liveAtmosphere?: LiveAtmosphereHourly;
  liveMarine?: LiveMarineHourly;
  horizonHours: 12 | 24 | 72;
  focus: ProjectionFocus;
  temperatureUnit: 'C' | 'F';
  waveThreshold: number;
  height?: number;
}) => {
  const option = useMemo(() => {
    const now = Date.now();
    const start = now - 24 * 60 * 60 * 1000;
    const end = now + horizonHours * 60 * 60 * 1000;
    const atmosphereTimes = liveAtmosphere?.time ?? [];
    const marineTimes = liveMarine?.time ?? [];
    const atmosphereByTime = new Map(atmosphereTimes.map((time, index) => [time, index]));
    const marineByTime = new Map(marineTimes.map((time, index) => [time, index]));
    const selected = Array.from(new Set([...atmosphereTimes, ...marineTimes]))
      .map((time) => ({ time, stamp: new Date(time).getTime() }))
      .filter(
        (point) => Number.isFinite(point.stamp) && point.stamp >= start && point.stamp <= end,
      );
    selected.sort((left, right) => left.stamp - right.stamp);
    const labels = selected.map((point) =>
      new Date(point.time).toLocaleString('fr-FR', {
        weekday: 'short',
        hour: '2-digit',
        timeZone: 'Africa/Casablanca',
      }),
    );
    const nowIndex = selected.reduce((best, point, index) => {
      if (best < 0) return index;
      return Math.abs(point.stamp - now) < Math.abs(selected[best].stamp - now) ? index : best;
    }, -1);
    const nowLabel = labels[Math.max(0, nowIndex)];
    const phaseGuides = labels.length
      ? {
          markArea: {
            silent: true,
            label: { show: true, position: 'insideTop', fontSize: 9, fontWeight: 800 },
            data: [
              [
                {
                  name: 'OBSERVÉ',
                  xAxis: labels[0],
                  itemStyle: { color: 'rgba(84,227,178,.035)' },
                  label: { color: '#71DDB0' },
                },
                { xAxis: nowLabel },
              ],
              [
                {
                  name: 'PRÉVU',
                  xAxis: nowLabel,
                  itemStyle: { color: 'rgba(141,145,255,.05)' },
                  label: { color: '#AAA7FF' },
                },
                { xAxis: labels[labels.length - 1] },
              ],
            ],
          },
          markLine: {
            silent: true,
            symbol: 'none',
            label: {
              show: true,
              formatter: 'MAINTENANT',
              color: '#06121C',
              backgroundColor: '#EAF7F8',
              borderRadius: 7,
              padding: [4, 7],
              fontSize: 8,
              fontWeight: 800,
            },
            lineStyle: { color: '#EAF7F8', width: 1.2, type: 'dashed', opacity: 0.72 },
            data: [{ xAxis: nowLabel }],
          },
        }
      : {};

    const splitSeries = (
      values: Array<number | null>,
      indexByTime: Map<string, number>,
      transform = (value: number | null) => value,
    ) => ({
      observed: selected.map((point) => {
        const index = indexByTime.get(point.time);
        return point.stamp <= now && index != null ? transform(values[index] ?? null) : null;
      }),
      forecast: selected.map((point) => {
        const index = indexByTime.get(point.time);
        return point.stamp >= now && index != null ? transform(values[index] ?? null) : null;
      }),
    });
    const splitMarineSeries = (values: Array<number | null>) => ({
      observed: selected.map((point) => {
        const index = marineByTime.get(point.time);
        return point.stamp <= now && index != null ? values[index] ?? null : null;
      }),
      forecast: selected.map((point) => {
        const index = marineByTime.get(point.time);
        return point.stamp >= now && index != null ? values[index] ?? null : null;
      }),
    });

    const temperature = splitSeries(
      liveAtmosphere?.temperature_2m ?? [],
      atmosphereByTime,
      (value) => convertTemperature(value, temperatureUnit),
    );
    const apparent = splitSeries(
      liveAtmosphere?.apparent_temperature ?? [],
      atmosphereByTime,
      (value) => convertTemperature(value, temperatureUnit),
    );
    const wave = splitMarineSeries(liveMarine?.wave_height ?? []);
    const swell = splitMarineSeries(liveMarine?.swell_wave_height ?? []);
    const temperatureAxis = axis(`°${temperatureUnit}`);
    const waveAxis = axis('m', focus === 'COMBINED' ? 'right' : 'left');
    const series: object[] = [];

    if (focus !== 'WAVES') {
      series.push(
        {
          ...line('Température observée', temperature.observed, pf.functional.amber, 0, false),
          lineStyle: { color: pf.functional.amber, width: 3 },
          ...phaseGuides,
        },
        {
          ...line('Température prévue', temperature.forecast, pf.functional.blue, 0, true),
          lineStyle: { color: pf.functional.blue, width: 3.2, type: 'dashed' },
          z: 5,
        },
      );
      if (focus === 'TEMPERATURE') {
        series.push({
          ...line('Ressenti prévu', apparent.forecast, pf.functional.purple),
          lineStyle: { color: pf.functional.purple, width: 2, type: 'dotted' },
        });
      }
    }

    if (focus !== 'TEMPERATURE') {
      const waveAxisIndex = focus === 'COMBINED' ? 1 : 0;
      series.push(
        {
          ...line('Vague observée', wave.observed, pf.functional.cyan, waveAxisIndex),
          lineStyle: { color: pf.functional.cyan, width: 3 },
          ...(focus === 'WAVES' ? phaseGuides : {}),
        },
        {
          ...line('Vague prévue', wave.forecast, pf.functional.blue, waveAxisIndex, true),
          lineStyle: { color: pf.functional.blue, width: 3.2, type: 'dashed' },
          z: 6,
          markLine: {
            silent: true,
            symbol: 'none',
            label: {
              formatter: `Seuil ${waveThreshold.toFixed(1)} m`,
              color: pf.functional.amber,
              fontSize: 10,
            },
            lineStyle: { color: pf.functional.amber, type: 'dotted', width: 1.8 },
            data: [{ yAxis: waveThreshold }],
          },
        },
      );
      if (focus === 'WAVES') {
        series.push({
          ...line('Houle prévue', swell.forecast, pf.functional.purple),
          lineStyle: { color: pf.functional.purple, width: 2, type: 'dotted' },
        });
      }
    }

    return {
      ...baseOption(labels),
      legend: {
        top: 0,
        left: 0,
        itemWidth: 19,
        itemHeight: 3,
        textStyle: { color: chartText, fontSize: 10 },
      },
      grid: { top: 62, left: 12, right: 16, bottom: 60, containLabel: true },
      yAxis:
        focus === 'COMBINED'
          ? [temperatureAxis, waveAxis]
          : [focus === 'TEMPERATURE' ? temperatureAxis : waveAxis],
      series,
    };
  }, [focus, horizonHours, liveAtmosphere, liveMarine, temperatureUnit, waveThreshold]);

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export const AtmosphereAnalyticsChart = ({
  live,
  forecast,
  mode,
  height = 330,
}: {
  live?: LiveAtmosphereHourly;
  forecast: MetoceanForecastPoint[];
  mode: ForecastMode;
  height?: number;
}) => {
  const option = useMemo(() => {
    if (mode === 'NEXT_24H') {
      const labels = (live?.time ?? []).map(formatHour);
      return {
        ...baseOption(labels),
        yAxis: [axis('°C'), axis('m/s', 'right')],
        series: [
          line('Température', live?.temperature_2m ?? [], pf.functional.amber, 0, true),
          line('Ressenti', live?.apparent_temperature ?? [], pf.functional.purple),
          line('Vent', live?.wind_speed_10m ?? [], pf.functional.cyan, 1),
        ],
      };
    }

    const temperature = groupForecast(forecast, 'temperature_2m');
    const wind = groupForecast(forecast, 'wind_speed_ms');
    const labels = (temperature.length ? temperature : wind).map((point) => `H+${point.horizon_h}`);
    return {
      ...baseOption(labels),
      yAxis: [axis('°C'), axis('m/s', 'right')],
      series: [
        line(
          'Température',
          temperature.map((point) => point.p50),
          pf.functional.amber,
          0,
          true,
        ),
        line(
          'Vent',
          wind.map((point) => point.p50),
          pf.functional.cyan,
          1,
        ),
      ],
    };
  }, [forecast, live, mode]);

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export const MarineAnalyticsChart = ({
  live,
  forecast,
  mode,
  height = 330,
}: {
  live?: LiveMarineHourly;
  forecast: MetoceanForecastPoint[];
  mode: ForecastMode;
  height?: number;
}) => {
  const option = useMemo(() => {
    if (mode === 'NEXT_24H') {
      const now = Date.now();
      const start = now - 24 * 60 * 60 * 1000;
      const end = now + 24 * 60 * 60 * 1000;
      const selected = (live?.time ?? [])
        .map((time, index) => ({ time, index, stamp: new Date(time).getTime() }))
        .filter((point) => point.stamp >= start && point.stamp <= end);
      const labels = selected.map((point) =>
        new Date(point.time).toLocaleString('fr-FR', {
          weekday: 'short',
          hour: '2-digit',
          timeZone: 'Africa/Casablanca',
        }),
      );
      const split = (values: Array<number | null>) => ({
        observed: selected.map((point) =>
          point.stamp <= now ? values[point.index] ?? null : null,
        ),
        forecast: selected.map((point) =>
          point.stamp >= now ? values[point.index] ?? null : null,
        ),
      });
      const wave = split(live?.wave_height ?? []);
      const swell = split(live?.swell_wave_height ?? []);
      const period = split(live?.wave_period ?? []);
      const nowIndex = selected.reduce((best, point, index) => {
        if (best < 0) return index;
        return Math.abs(point.stamp - now) < Math.abs(selected[best].stamp - now) ? index : best;
      }, -1);
      const nowLabel = labels[Math.max(0, nowIndex)];
      const marinePhases = labels.length
        ? {
            markArea: {
              silent: true,
              label: { show: true, position: 'insideTop', fontSize: 9, fontWeight: 800 },
              data: [
                [
                  {
                    name: 'OBSERVÉ',
                    xAxis: labels[0],
                    itemStyle: { color: 'rgba(84,227,178,.04)' },
                    label: { color: '#71DDB0' },
                  },
                  { xAxis: nowLabel },
                ],
                [
                  {
                    name: 'PRÉVU',
                    xAxis: nowLabel,
                    itemStyle: { color: 'rgba(141,145,255,.055)' },
                    label: { color: '#AAA7FF' },
                  },
                  { xAxis: labels[labels.length - 1] },
                ],
              ],
            },
            markLine: {
              silent: true,
              symbol: 'none',
              label: {
                formatter: 'MAINTENANT',
                color: '#06121C',
                backgroundColor: '#EAF7F8',
                borderRadius: 7,
                padding: [4, 7],
                fontSize: 8,
                fontWeight: 800,
              },
              lineStyle: { color: '#EAF7F8', type: 'dashed', opacity: 0.7 },
              data: [{ xAxis: nowLabel }],
            },
          }
        : {};
      return {
        ...baseOption(labels),
        yAxis: [axis('m'), axis('s', 'right')],
        series: [
          {
            ...line('Vague · observée', wave.observed, '#35D7F2', 0, true),
            lineStyle: { color: '#35D7F2', width: 3.2 },
            ...marinePhases,
          },
          {
            ...line('Vague · prévue', wave.forecast, '#49A7FF', 0, true),
            lineStyle: { color: '#49A7FF', width: 3.2, type: 'dashed' },
          },
          {
            ...line('Houle · observée', swell.observed, '#54E3B2'),
            lineStyle: { color: '#54E3B2', width: 2.2 },
          },
          {
            ...line('Houle · prévue', swell.forecast, '#9A91FF'),
            lineStyle: { color: '#9A91FF', width: 2.4, type: 'dashed' },
          },
          {
            ...line('Période · observée', period.observed, '#F4C76A', 1),
            lineStyle: { color: '#F4C76A', width: 1.9 },
          },
          {
            ...line('Période · prévue', period.forecast, '#FF9F68', 1),
            lineStyle: { color: '#FF9F68', width: 2.1, type: 'dashed' },
          },
        ],
      };
    }

    const wave = groupForecast(forecast, 'wave_height_m');
    const period = groupForecast(forecast, 'wave_period_s');
    const labels = (wave.length ? wave : period).map((point) => `H+${point.horizon_h}`);
    return {
      ...baseOption(labels),
      yAxis: [axis('m'), axis('s', 'right')],
      series: [
        line(
          'Vague',
          wave.map((point) => point.p50),
          pf.functional.blue,
          0,
          true,
        ),
        line(
          'Période',
          period.map((point) => point.p50),
          pf.functional.purple,
          1,
        ),
      ],
    };
  }, [forecast, live, mode]);

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export const KpiSparkline = ({ data, color }: { data: Array<number | null>; color: string }) => {
  const option = useMemo(
    () => ({
      animationDuration: 1150,
      animationEasing: 'cubicOut',
      grid: { top: 5, left: 1, right: 1, bottom: 3 },
      xAxis: { type: 'category', show: false, data: data.map((_, index) => index) },
      yAxis: { type: 'value', show: false, scale: true },
      series: [
        {
          ...line('', data, color, 0, true),
          lineStyle: { color, width: 3, cap: 'round', join: 'round' },
          symbolSize: 6,
          showSymbol: false,
        },
      ],
    }),
    [color, data],
  );

  return (
    <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height: 58, flexShrink: 0 }} />
  );
};

export const MarineCompositionDonut = ({
  windWaveHeight,
  swellWaveHeight,
  totalWaveHeight,
  height = 220,
}: {
  windWaveHeight?: number | null;
  swellWaveHeight?: number | null;
  totalWaveHeight?: number | null;
  height?: number;
}) => {
  const option = useMemo(() => {
    const windEnergy = Math.pow(Math.max(windWaveHeight ?? 0, 0), 2);
    const swellEnergy = Math.pow(Math.max(swellWaveHeight ?? 0, 0), 2);
    const hasEnergy = windEnergy + swellEnergy > 0;
    const data = hasEnergy
      ? [
          { name: 'Mer du vent', value: windEnergy, itemStyle: { color: '#00E5C7' } },
          { name: 'Houle', value: swellEnergy, itemStyle: { color: '#8B7CFF' } },
        ]
      : [{ name: 'Indisponible', value: 1, itemStyle: { color: 'rgba(255,255,255,0.10)' } }];

    return {
      animationDuration: 900,
      animationEasing: 'cubicOut',
      title: {
        text: totalWaveHeight == null ? '—' : `${totalWaveHeight.toFixed(2)} m`,
        subtext: 'VAGUE TOTALE',
        left: 'center',
        top: '37%',
        textStyle: {
          color: pf.text.primary,
          fontSize: 22,
          fontWeight: 600,
          fontFamily: 'Inter, sans-serif',
        },
        subtextStyle: { color: chartText, fontSize: 9, fontWeight: 700 },
      },
      tooltip: {
        trigger: 'item',
        formatter: (params: { name: string; percent: number }) =>
          params.name === 'Indisponible'
            ? 'Composition indisponible'
            : `${params.name}<br/><b>${params.percent.toFixed(0)} %</b> de l'énergie`,
        backgroundColor: 'rgba(13,16,27,0.96)',
        borderColor: 'rgba(255,255,255,0.12)',
        textStyle: { color: '#F1F4F9' },
      },
      series: [
        {
          type: 'pie',
          radius: ['66%', '86%'],
          center: ['50%', '50%'],
          startAngle: 90,
          clockwise: true,
          avoidLabelOverlap: true,
          label: { show: false },
          labelLine: { show: false },
          itemStyle: { borderColor: '#0A0C14', borderWidth: 4, borderRadius: 8 },
          emphasis: { scale: true, scaleSize: 5 },
          data,
        },
      ],
    };
  }, [swellWaveHeight, totalWaveHeight, windWaveHeight]);

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};

export const ExposureBarChart = ({
  liveAtmosphere,
  liveMarine,
  forecast,
  mode,
  height = 280,
}: {
  liveAtmosphere?: LiveAtmosphereHourly;
  liveMarine?: LiveMarineHourly;
  forecast: MetoceanForecastPoint[];
  mode: ForecastMode;
  height?: number;
}) => {
  const option = useMemo(() => {
    const labels: string[] = [];
    const values: number[] = [];

    if (mode === 'NEXT_24H') {
      const times = liveAtmosphere?.time ?? liveMarine?.time ?? [];
      const wave = liveMarine?.wave_height ?? [];
      const gust = liveAtmosphere?.wind_gusts_10m ?? [];
      const rain = liveAtmosphere?.precipitation_probability ?? [];
      times.forEach((time, index) => {
        if (index % 3 !== 0) return;
        const wavePressure = ((wave[index] ?? 0) / 2.5) * 100;
        const gustPressure = ((gust[index] ?? 0) / 20) * 100;
        const rainPressure = rain[index] ?? 0;
        labels.push(formatHour(time));
        values.push(Math.round(Math.min(100, Math.max(wavePressure, gustPressure, rainPressure))));
      });
    } else {
      const wave = groupForecast(forecast, 'wave_height_m');
      const wind = groupForecast(forecast, 'wind_speed_ms');
      const horizons = Array.from(new Set([...wave, ...wind].map((point) => point.horizon_h))).sort(
        (a, b) => a - b,
      );
      horizons.forEach((horizon) => {
        const waveValue = wave.find((point) => point.horizon_h === horizon)?.p50 ?? 0;
        const windValue = wind.find((point) => point.horizon_h === horizon)?.p50 ?? 0;
        labels.push(`H+${horizon}`);
        values.push(
          Math.round(Math.min(100, Math.max((waveValue / 2.5) * 100, (windValue / 20) * 100))),
        );
      });
    }

    return {
      animationDuration: 900,
      animationEasing: 'cubicOut',
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(255,255,255,0.04)' } },
        formatter: (params: Array<{ axisValue: string; value: number }>) =>
          `${params[0]?.axisValue ?? ''}<br/>Indice de vigilance <b>${params[0]?.value ?? 0}/100</b>`,
        backgroundColor: 'rgba(13,16,27,0.96)',
        borderColor: 'rgba(255,255,255,0.12)',
        textStyle: { color: '#F1F4F9' },
      },
      grid: { top: 18, left: 10, right: 8, bottom: 28, containLabel: true },
      xAxis: {
        type: 'category',
        data: labels,
        axisTick: { show: false },
        axisLine: { lineStyle: { color: 'rgba(255,255,255,0.10)' } },
        axisLabel: { color: chartText, fontSize: 10, interval: 0 },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        axisLabel: { color: chartText, fontSize: 10, formatter: '{value}' },
        splitLine: { lineStyle: { color: chartGrid } },
      },
      series: [
        {
          name: 'Vigilance',
          type: 'bar',
          data: values,
          barMaxWidth: 32,
          itemStyle: {
            borderRadius: [6, 6, 2, 2],
            color: (params: { value: number }) => {
              if (params.value >= 75) return '#FF6B6B';
              if (params.value >= 50) return '#FFB454';
              return '#00E5C7';
            },
          },
        },
      ],
    };
  }, [forecast, liveAtmosphere, liveMarine, mode]);

  return <ReactEchart echarts={echarts} option={option} sx={{ width: 1, height }} />;
};
