import { Box, Chip, CircularProgress, Stack, Typography } from '@mui/material';
import type { ReactNode } from 'react';
import IconifyIcon from 'components/base/IconifyIcon';
import { formatTimestamp } from 'helpers/maritime';
import { useReplay } from 'providers/ReplayProvider';

interface MaritimePageHeaderProps {
  title: string;
  subtitle: string;
  icon: string;
  actions?: ReactNode;
}

const MaritimePageHeader = ({ title, subtitle, icon, actions }: MaritimePageHeaderProps) => {
  const { asOf, snapshot, refreshing } = useReplay();

  return (
    <Stack
      direction={{ xs: 'column', lg: 'row' }}
      justifyContent="space-between"
      alignItems={{ xs: 'stretch', lg: 'center' }}
      gap={2}
    >
      <Stack direction="row" alignItems="center" gap={1.5} minWidth={0}>
        <Box
          sx={{
            display: 'grid',
            placeItems: 'center',
            flex: '0 0 auto',
            width: 42,
            height: 42,
            borderRadius: 2,
            color: '#0F766E',
            bgcolor: '#DDF4F1',
          }}
        >
          <IconifyIcon icon={icon} sx={{ fontSize: 24 }} />
        </Box>
        <Box minWidth={0}>
          <Typography variant="h3">{title}</Typography>
          <Typography variant="body2" color="text.secondary" mt={0.25}>
            {subtitle}
          </Typography>
        </Box>
      </Stack>
      <Stack direction="row" alignItems="center" gap={1} flexWrap="wrap">
        {actions}
        {refreshing && <CircularProgress size={17} />}
        <Chip
          size="small"
          icon={<IconifyIcon icon="material-symbols:schedule-rounded" />}
          label={formatTimestamp(snapshot?.resolved_as_of ?? asOf)}
          variant="outlined"
          sx={{ bgcolor: 'background.paper' }}
        />
      </Stack>
    </Stack>
  );
};

export default MaritimePageHeader;
