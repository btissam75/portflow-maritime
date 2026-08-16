import { AppBar, Box, Chip, IconButton, Stack, Toolbar, Tooltip, Typography } from '@mui/material';
import IconifyIcon from 'components/base/IconifyIcon';
import { useEffect, useMemo, useState } from 'react';
import { useLocation } from 'react-router-dom';
import paths from 'routes/paths';
import { portflowPalette as pf } from 'theme/portflowPalette';
import ElevationScroll from './ElevationScroll';

const Topbar = () => {
  const location = useLocation();
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 30_000);
    return () => window.clearInterval(timer);
  }, []);

  const page = location.pathname.startsWith(paths.capacity)
    ? { title: 'Escales & capacité', context: 'Priorisation temporelle des revues', icon: 'lucide:ship' }
    : location.pathname.startsWith(paths.weather)
      ? { title: 'Météo & état de mer', context: 'Conditions, prévisions et vigilance', icon: 'lucide:cloud-sun' }
      : { title: 'Control Tower', context: 'Supervision opérationnelle unifiée', icon: 'lucide:layout-dashboard' };

  const localDate = useMemo(() => new Intl.DateTimeFormat('fr-FR', {
    weekday: 'short', day: '2-digit', month: 'short', timeZone: 'Africa/Casablanca',
  }).format(now), [now]);
  const localTime = useMemo(() => new Intl.DateTimeFormat('fr-FR', {
    hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'Africa/Casablanca',
  }).format(now), [now]);

  return (
    <ElevationScroll>
      <AppBar position="fixed" sx={{ left: { xs: 0, md: 64, xl: 208 }, width: { xs: 1, md: 'calc(100% - 64px)', xl: 'calc(100% - 208px)' }, borderBottom: `1px solid ${pf.structure.border}`, boxShadow: 'none', bgcolor: 'rgba(7,23,34,0.88)', backdropFilter: 'blur(14px)', '@supports not (backdrop-filter: blur(14px))': { bgcolor: pf.background.navigation } }}>
        <Toolbar sx={{ minHeight: '64px !important', gap: 1.25, px: { xs: 1.5, sm: 2.5 } }}>
          <Box sx={{ display: { xs: 'grid', md: 'none' }, width: 36, height: 36, placeItems: 'center', color: pf.functional.cyan, bgcolor: pf.functional.cyanSoft, border: `1px solid ${pf.functional.cyan}45`, borderRadius: '8px' }}>
            <IconifyIcon icon="lucide:anchor" sx={{ fontSize: 18 }} />
          </Box>

          <Stack direction="row" alignItems="center" gap={1} minWidth={0}>
            <Box sx={{ display: { xs: 'none', sm: 'grid' }, width: 34, height: 34, placeItems: 'center', color: pf.functional.cyan, bgcolor: pf.background.panelRaised, border: `1px solid ${pf.structure.border}`, borderRadius: '8px' }}>
              <IconifyIcon icon={page.icon} sx={{ fontSize: 17 }} />
            </Box>
            <Box minWidth={0}>
              <Typography noWrap sx={{ color: pf.text.primary, fontSize: 14, lineHeight: '19px', fontWeight: 650 }}>{page.title}</Typography>
              <Typography noWrap sx={{ display: { xs: 'none', sm: 'block' }, color: pf.text.tertiary, fontSize: 11, lineHeight: '16px' }}>{page.context}</Typography>
            </Box>
          </Stack>

          <Stack direction="row" alignItems="center" gap={0.85} ml="auto">
            <Chip icon={<IconifyIcon icon="lucide:radio" />} label="LIVE" size="small" sx={{ display: { xs: 'none', sm: 'flex' }, height: 28, color: pf.functional.green, bgcolor: pf.functional.greenSoft, border: `1px solid ${pf.functional.green}45`, '& .MuiChip-icon': { color: pf.functional.green } }} />
            <Box sx={{ display: { xs: 'none', sm: 'block' }, textAlign: 'right', minWidth: 92 }}>
              <Typography sx={{ color: pf.text.primary, fontSize: 12, lineHeight: '16px', fontWeight: 600 }}>{localTime}</Typography>
              <Typography sx={{ color: pf.text.tertiary, fontSize: 11, lineHeight: '15px', textTransform: 'capitalize' }}>{localDate}</Typography>
            </Box>
            <Tooltip title="Notifications opérationnelles">
              <IconButton aria-label="Notifications opérationnelles" sx={{ width: 36, height: 36, color: pf.text.secondary, border: `1px solid ${pf.structure.border}`, borderRadius: '8px' }}>
                <IconifyIcon icon="lucide:bell" sx={{ fontSize: 17 }} />
              </IconButton>
            </Tooltip>
          </Stack>
        </Toolbar>
      </AppBar>
    </ElevationScroll>
  );
};

export default Topbar;
