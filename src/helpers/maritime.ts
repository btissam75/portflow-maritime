import type { PortCallStatus } from 'types/replay';

export const HOUR_MS = 60 * 60 * 1000;
export const HORIZONS = [6, 12, 24];

const dateTimeFormatter = new Intl.DateTimeFormat('fr-FR', {
  dateStyle: 'medium',
  timeStyle: 'short',
  timeZone: 'UTC',
});

const shortDateFormatter = new Intl.DateTimeFormat('fr-FR', {
  day: '2-digit',
  month: 'short',
  hour: '2-digit',
  timeZone: 'UTC',
});

export const numberFormatter = new Intl.NumberFormat('fr-FR', {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

export const integerFormatter = new Intl.NumberFormat('fr-FR', {
  maximumFractionDigits: 0,
});

export const formatTimestamp = (value: string | number | Date | null | undefined) =>
  value ? `${dateTimeFormatter.format(new Date(value))} UTC` : 'Indisponible';

export const formatShortTimestamp = (value: string | number | Date) =>
  shortDateFormatter.format(new Date(value));

export const formatDuration = (hours: number | null | undefined) => {
  if (hours == null) return '—';
  const sign = hours > 0 ? '+' : '';
  return `${sign}${numberFormatter.format(hours)} h`;
};

export const statusLabels: Record<PortCallStatus, string> = {
  EXPECTED: 'Attendue',
  OVERDUE: 'En dépassement',
  ARRIVED: 'Arrivée',
  BERTHED: 'À quai',
  DEPARTED: 'Partie',
};

export const statusColors: Record<
  PortCallStatus,
  'default' | 'warning' | 'info' | 'success' | 'error'
> = {
  EXPECTED: 'info',
  OVERDUE: 'error',
  ARRIVED: 'success',
  BERTHED: 'warning',
  DEPARTED: 'default',
};
