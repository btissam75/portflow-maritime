import { Avatar, Box, Stack, Tooltip, Typography } from '@mui/material';
import IconifyIcon from 'components/base/IconifyIcon';
import { Link, useLocation } from 'react-router-dom';
import paths from 'routes/paths';
import { portflowPalette as pf } from 'theme/portflowPalette';

const navigation = [
  {
    group: 'SUPERVISION',
    items: [
      {
        label: 'Vue globale',
        shortLabel: 'Global',
        path: paths.overview,
        icon: 'lucide:layout-dashboard',
      },
      {
        label: 'Unités prioritaires',
        shortLabel: 'Unités',
        path: paths.units,
        icon: 'lucide:container',
      },
      { label: 'Flux métier', shortLabel: 'Flux', path: paths.process, icon: 'lucide:git-branch' },
      {
        label: 'Prévisions',
        shortLabel: 'Prévisions',
        path: paths.forecast,
        icon: 'lucide:chart-no-axes-combined',
      },
    ],
  },
  {
    group: 'DÉCISION',
    items: [
      { label: 'Alertes', shortLabel: 'Alertes', path: paths.alerts, icon: 'lucide:shield-alert' },
      {
        label: 'Décisions',
        shortLabel: 'Décisions',
        path: paths.decisions,
        icon: 'lucide:list-checks',
      },
      {
        label: 'Simulation',
        shortLabel: 'What-if',
        path: paths.simulation,
        icon: 'lucide:flask-conical',
      },
    ],
  },
  {
    group: 'TERRITOIRE',
    items: [
      {
        label: 'Approches navires',
        shortLabel: 'Navires',
        path: paths.vessels,
        icon: 'lucide:ship',
      },
      {
        label: 'Météo & état de mer',
        shortLabel: 'Météo',
        path: paths.weather,
        icon: 'lucide:cloud-sun',
      },
      {
        label: 'Escales & capacité',
        shortLabel: 'Escales',
        path: paths.capacity,
        icon: 'lucide:calendar-range',
      },
    ],
  },
  {
    group: 'GOUVERNANCE',
    items: [
      {
        label: 'Qualité des données',
        shortLabel: 'Qualité',
        path: paths.quality,
        icon: 'lucide:database-zap',
      },
      {
        label: 'Journal d’audit',
        shortLabel: 'Audit',
        path: paths.audit,
        icon: 'lucide:scroll-text',
      },
      {
        label: 'Rapports',
        shortLabel: 'Rapports',
        path: paths.reports,
        icon: 'lucide:file-chart-column',
      },
    ],
  },
] as const;

type NavigationItem = {
  label: string;
  shortLabel: string;
  path: string;
  icon: string;
};

const mobileItems: NavigationItem[] = navigation.reduce<NavigationItem[]>(
  (items, section) => [...items, ...section.items],
  [],
);

const NavigationRail = () => {
  const location = useLocation();

  return (
    <>
      <Box
        component="aside"
        sx={{
          display: { xs: 'none', md: 'flex' },
          position: 'fixed',
          inset: '0 auto 0 0',
          zIndex: 1300,
          width: 78,
          flexDirection: 'column',
          bgcolor: pf.background.navigation,
          borderRight: `1px solid ${pf.structure.border}`,
          overflow: 'hidden',
          boxShadow: '12px 0 34px rgba(0,0,0,.18)',
        }}
      >
        <Stack
          direction="row"
          alignItems="center"
          gap={1.1}
          sx={{
            height: 84,
            px: 0,
            justifyContent: 'center',
            borderBottom: `1px solid ${pf.structure.border}`,
          }}
        >
          <Box
            sx={{
              width: 44,
              height: 44,
              display: 'grid',
              placeItems: 'center',
              color: pf.functional.cyan,
              bgcolor: pf.functional.cyanSoft,
              border: `1px solid ${pf.functional.cyan}55`,
              borderRadius: '13px',
              flexShrink: 0,
            }}
          >
            <IconifyIcon icon="lucide:waves" sx={{ fontSize: 22 }} />
          </Box>
        </Stack>

        <Box
          component="nav"
          sx={{
            flex: 1,
            minHeight: 0,
            overflowY: 'auto',
            px: 1,
            py: 1.3,
            '&::-webkit-scrollbar': { width: 3 },
            '&::-webkit-scrollbar-thumb': { bgcolor: pf.structure.border, borderRadius: 4 },
          }}
        >
          {navigation.map((section) => (
            <Box key={section.group} mb={0.45}>
              <Typography
                sx={{
                  display: 'none',
                  px: 1.1,
                  py: 0.5,
                  color: pf.text.disabled,
                  fontSize: 7,
                  fontWeight: 900,
                  letterSpacing: '.13em',
                }}
              >
                {section.group}
              </Typography>
              <Stack gap={0.55}>
                {section.items.map((item) => {
                  const active =
                    location.pathname === item.path ||
                    location.pathname.startsWith(`${item.path}/`);
                  return (
                    <Tooltip key={item.path} title={item.label} placement="right" arrow>
                      <Box
                        component={Link}
                        to={item.path}
                        aria-current={active ? 'page' : undefined}
                        sx={{
                          position: 'relative',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: 1,
                          height: 42,
                          px: 0,
                          color: active ? pf.text.primary : pf.text.tertiary,
                          background: active
                            ? 'linear-gradient(90deg, rgba(85,214,194,.15), rgba(165,139,250,.035))'
                            : 'transparent',
                          borderRadius: '11px',
                          textDecoration: 'none',
                          transition: 'all 140ms ease',
                          '&::before': {
                            content: '""',
                            position: 'absolute',
                            left: -8,
                            width: 3,
                            height: active ? 26 : 0,
                            bgcolor: pf.functional.cyan,
                          },
                          '&:hover': {
                            bgcolor: 'rgba(85,214,194,.07)',
                            color: pf.text.primary,
                            transform: 'translateY(-1px)',
                          },
                          '&:focus-visible': {
                            outline: `2px solid ${pf.functional.cyan}`,
                            outlineOffset: -2,
                          },
                        }}
                      >
                        <IconifyIcon
                          icon={item.icon}
                          sx={{
                            color: active ? pf.functional.cyan : 'inherit',
                            fontSize: 19,
                            flexShrink: 0,
                          }}
                        />
                      </Box>
                    </Tooltip>
                  );
                })}
              </Stack>
            </Box>
          ))}
        </Box>

        <Stack p={1} sx={{ borderTop: `1px solid ${pf.structure.border}` }}>
          <Stack direction="row" alignItems="center" justifyContent="center" gap={1}>
            <Avatar
              sx={{
                width: 31,
                height: 31,
                bgcolor: pf.functional.cyanSoft,
                color: pf.functional.cyan,
                border: `1px solid ${pf.functional.cyan}45`,
                fontSize: 9,
              }}
            >
              OP
            </Avatar>
          </Stack>
        </Stack>
      </Box>

      <Box
        component="nav"
        aria-label="Navigation principale"
        sx={{
          display: { xs: 'flex', md: 'none' },
          position: 'fixed',
          zIndex: 1300,
          left: 8,
          right: 8,
          bottom: 8,
          height: 61,
          bgcolor: 'rgba(13,15,18,.96)',
          border: `1px solid ${pf.structure.border}`,
          borderRadius: '12px',
          boxShadow: '0 16px 40px rgba(0,0,0,0.4)',
          backdropFilter: 'blur(14px)',
          overflowX: 'auto',
          overflowY: 'hidden',
          '&::-webkit-scrollbar': { display: 'none' },
        }}
      >
        {mobileItems.map((item) => {
          const active =
            location.pathname === item.path || location.pathname.startsWith(`${item.path}/`);
          return (
            <Stack
              key={item.path}
              component={Link}
              to={item.path}
              alignItems="center"
              justifyContent="center"
              gap={0.25}
              sx={{
                minWidth: 72,
                color: active ? pf.functional.cyan : pf.text.tertiary,
                textDecoration: 'none',
                bgcolor: active ? pf.functional.cyanSoft : 'transparent',
                borderTop: active ? `2px solid ${pf.functional.cyan}` : '2px solid transparent',
              }}
            >
              <IconifyIcon icon={item.icon} sx={{ fontSize: 17 }} />
              <Typography sx={{ color: 'inherit', fontSize: 8.5, whiteSpace: 'nowrap' }}>
                {item.shortLabel}
              </Typography>
            </Stack>
          );
        })}
      </Box>
    </>
  );
};

export default NavigationRail;
