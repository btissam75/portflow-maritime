import { Avatar, Box, Stack, Tooltip, Typography } from '@mui/material';
import IconifyIcon from 'components/base/IconifyIcon';
import { Link, useLocation } from 'react-router-dom';
import paths from 'routes/paths';
import { portflowPalette as pf } from 'theme/portflowPalette';

const navigation = [
  { label: 'Control Tower', shortLabel: 'Contrôle', path: paths.controlTower, icon: 'lucide:layout-dashboard' },
  { label: 'Météo & état de mer', shortLabel: 'Météo', path: paths.weather, icon: 'lucide:cloud-sun' },
  { label: 'Escales & capacité', shortLabel: 'Escales', path: paths.capacity, icon: 'lucide:ship' },
] as const;

const NavigationRail = () => {
  const location = useLocation();

  return (
    <>
      <Box component="aside" sx={{ display: { xs: 'none', md: 'flex' }, position: 'fixed', inset: '0 auto 0 0', zIndex: 1300, width: { md: 64, xl: 208 }, flexDirection: 'column', bgcolor: pf.background.navigation, borderRight: `1px solid ${pf.structure.border}`, overflow: 'hidden', transition: 'width 220ms cubic-bezier(0.2,0.8,0.2,1)' }}>
        <Stack direction="row" alignItems="center" gap={1.1} sx={{ height: 64, px: { md: 1.4, xl: 2 }, borderBottom: `1px solid ${pf.structure.border}` }}>
          <Box sx={{ width: 36, height: 36, display: 'grid', placeItems: 'center', color: pf.functional.cyan, bgcolor: pf.functional.cyanSoft, border: `1px solid ${pf.functional.cyan}55`, borderRadius: '8px', flexShrink: 0 }}>
            <IconifyIcon icon="lucide:anchor" sx={{ fontSize: 19 }} />
          </Box>
          <Box sx={{ display: { md: 'none', xl: 'block' }, minWidth: 0 }}>
            <Typography sx={{ color: pf.text.primary, fontSize: 14, fontWeight: 700 }}>PORTFLOW</Typography>
            <Typography sx={{ color: pf.text.tertiary, fontSize: 11 }}>TANGER MED</Typography>
          </Box>
        </Stack>

        <Stack component="nav" gap={0.45} px={{ md: 0.75, xl: 1 }} py={1.5}>
          {navigation.map((item) => {
            const active = location.pathname.startsWith(item.path);
            return (
              <Tooltip key={item.path} title={item.label} placement="right" arrow>
                <Box component={Link} to={item.path} aria-current={active ? 'page' : undefined} sx={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: { md: 'center', xl: 'flex-start' }, width: 1, height: 40, px: { md: 0, xl: 1.15 }, color: active ? pf.text.primary : pf.text.secondary, background: active ? 'linear-gradient(90deg, rgba(54,214,207,0.18), rgba(73,167,255,0.05))' : 'transparent', borderRadius: '8px', textDecoration: 'none', transition: 'background-color 140ms ease, color 140ms ease, transform 140ms ease', '&::before': { content: '""', position: 'absolute', left: 0, width: 2, height: active ? 24 : 0, bgcolor: pf.functional.cyan }, '&:hover': { bgcolor: 'rgba(54,214,207,0.07)', color: pf.text.primary, transform: 'translateX(2px)' }, '&:focus-visible': { outline: `2px solid ${pf.functional.blue}`, outlineOffset: -2 } }}>
                  <IconifyIcon icon={item.icon} sx={{ color: active ? pf.functional.cyan : 'inherit', fontSize: 17, flexShrink: 0 }} />
                  <Typography noWrap sx={{ display: { md: 'none', xl: 'block' }, ml: 1, color: 'inherit', fontSize: 13, fontWeight: active ? 600 : 400 }}>{item.label}</Typography>
                </Box>
              </Tooltip>
            );
          })}
        </Stack>

        <Stack mt="auto" p={{ md: 1, xl: 1.5 }}>
          <Stack direction="row" alignItems="center" justifyContent={{ md: 'center', xl: 'flex-start' }} gap={1}>
            <Avatar sx={{ width: 34, height: 34, bgcolor: pf.functional.cyanSoft, color: pf.functional.cyan, border: `1px solid ${pf.functional.cyan}45`, fontSize: 11 }}>OP</Avatar>
            <Box sx={{ display: { md: 'none', xl: 'block' } }}>
              <Typography sx={{ color: pf.text.primary, fontSize: 12, fontWeight: 600 }}>Opérateur salle</Typography>
              <Typography sx={{ color: pf.text.tertiary, fontSize: 11 }}>Quart B</Typography>
            </Box>
          </Stack>
        </Stack>
      </Box>

      <Box component="nav" aria-label="Navigation principale" sx={{ display: { xs: 'grid', md: 'none' }, position: 'fixed', zIndex: 1300, left: 10, right: 10, bottom: 10, gridTemplateColumns: 'repeat(3,minmax(0,1fr))', bgcolor: 'rgba(7,23,34,0.96)', border: `1px solid ${pf.structure.border}`, borderRadius: '8px', boxShadow: '0 16px 40px rgba(0,0,0,0.34)', backdropFilter: 'blur(14px)', overflow: 'hidden' }}>
        {navigation.map((item) => {
          const active = location.pathname.startsWith(item.path);
          return (
            <Stack key={item.path} component={Link} to={item.path} alignItems="center" justifyContent="center" gap={0.25} sx={{ minHeight: 58, color: active ? pf.functional.cyan : pf.text.tertiary, textDecoration: 'none', bgcolor: active ? pf.functional.cyanSoft : 'transparent', borderTop: active ? `2px solid ${pf.functional.cyan}` : '2px solid transparent' }}>
              <IconifyIcon icon={item.icon} sx={{ fontSize: 18 }} />
              <Typography sx={{ color: 'inherit', fontSize: 11 }}>{item.shortLabel}</Typography>
            </Stack>
          );
        })}
      </Box>
    </>
  );
};

export default NavigationRail;
