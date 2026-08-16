import { Box, Stack, Typography } from '@mui/material';
import IconifyIcon from 'components/base/IconifyIcon';
import { portflowPalette as pf } from 'theme/portflowPalette';

interface PortflowKpiProps {
  label: string;
  value: string;
  unit?: string;
  detail: string;
  icon: string;
  accent?: string;
  delay?: number;
}

const PortflowKpi = ({ label, value, unit, detail, icon, accent = pf.functional.cyan, delay = 0 }: PortflowKpiProps) => (
  <Box
    sx={{
      position: 'relative',
      minWidth: 0,
      height: 108,
      px: 1.6,
      py: 1.35,
      bgcolor: pf.background.panel,
      border: `1px solid ${pf.structure.border}`,
      borderRadius: '8px',
      overflow: 'hidden',
      animation: 'pfKpiIn 280ms cubic-bezier(0.16,1,0.3,1) both',
      animationDelay: `${delay}ms`,
      '@keyframes pfKpiIn': {
        from: { opacity: 0, transform: 'translateY(8px)' },
        to: { opacity: 1, transform: 'translateY(0)' },
      },
      '@media (prefers-reduced-motion: reduce)': {
        animation: 'none',
      },
    }}
  >
    <Stack direction="row" justifyContent="space-between" gap={1}>
      <Box minWidth={0}>
        <Typography noWrap sx={{ color: pf.text.secondary, fontSize: 12, lineHeight: '18px', fontWeight: 600 }}>
          {label}
        </Typography>
        <Stack direction="row" alignItems="baseline" gap={0.55} mt={0.4}>
          <Typography sx={{ color: pf.text.primary, fontSize: { xs: 25, xl: 30 }, lineHeight: '34px', fontWeight: 700 }}>
            {value}
          </Typography>
          {unit && <Typography sx={{ color: pf.text.secondary, fontSize: 12 }}>{unit}</Typography>}
        </Stack>
      </Box>
      <Box sx={{ width: 34, height: 34, display: 'grid', placeItems: 'center', flexShrink: 0, color: accent, bgcolor: `${accent}18`, border: `1px solid ${accent}42`, borderRadius: '8px' }}>
        <IconifyIcon icon={icon} sx={{ fontSize: 18 }} />
      </Box>
    </Stack>
    <Stack direction="row" alignItems="center" gap={0.7} mt={0.55}>
      <Box sx={{ width: 5, height: 5, borderRadius: '50%', bgcolor: accent }} />
      <Typography noWrap sx={{ color: pf.text.tertiary, fontSize: 12, lineHeight: '18px' }}>{detail}</Typography>
    </Stack>
  </Box>
);

export default PortflowKpi;
