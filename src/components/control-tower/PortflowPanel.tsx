import { Box, Paper, Stack, Typography } from '@mui/material';
import type { ReactNode } from 'react';
import { portflowPalette as pf } from 'theme/portflowPalette';

interface PortflowPanelProps {
  title: string;
  eyebrow?: string;
  action?: ReactNode;
  children: ReactNode;
  height?: number | string;
  sx?: object;
}

const PortflowPanel = ({ title, eyebrow, action, children, height, sx }: PortflowPanelProps) => (
  <Paper sx={{ minWidth: 0, height, overflow: 'hidden', borderRadius: '8px', ...sx }}>
    <Stack
      direction="row"
      alignItems="center"
      justifyContent="space-between"
      gap={1}
      sx={{ minHeight: 46, px: 1.75, borderBottom: `1px solid ${pf.structure.divider}` }}
    >
      <Box minWidth={0}>
        {eyebrow && (
          <Typography sx={{ color: pf.functional.cyan, fontSize: 11, lineHeight: '16px', fontWeight: 600, textTransform: 'uppercase' }}>
            {eyebrow}
          </Typography>
        )}
        <Typography noWrap sx={{ color: pf.text.primary, fontSize: 16, lineHeight: '22px', fontWeight: 650 }}>
          {title}
        </Typography>
      </Box>
      {action}
    </Stack>
    {children}
  </Paper>
);

export default PortflowPanel;
