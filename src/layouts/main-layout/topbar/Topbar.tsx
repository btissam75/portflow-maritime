import { AppBar, Box, Chip, Stack, Toolbar, Typography } from '@mui/material';
import IconifyIcon from 'components/base/IconifyIcon';
import { useEffect, useMemo, useState } from 'react';
import { useLocation } from 'react-router-dom';
import paths from 'routes/paths';
import { portflowPalette as pf } from 'theme/portflowPalette';
import ElevationScroll from './ElevationScroll';
import ControlTowerActions from './ControlTowerActions';

const pageDefinitions = [
  [paths.overview, 'Control Tower', 'Situation opérationnelle globale', 'lucide:layout-dashboard'],
  [
    paths.units,
    'Unités prioritaires',
    'File intelligente et fiches explicables',
    'lucide:container',
  ],
  [
    paths.process,
    'Flux métier',
    'Progression, charge et localisation interne',
    'lucide:git-branch',
  ],
  [
    paths.forecast,
    'Prévisions',
    'Anticipation multi-horizons et capacité',
    'lucide:chart-no-axes-combined',
  ],
  [paths.alerts, 'Alertes', 'Risques consolidés et actions proposées', 'lucide:shield-alert'],
  [paths.decisions, 'Décisions', 'Pilotage du traitement de bout en bout', 'lucide:list-checks'],
  [paths.vessels, 'Approches navires', 'Trafic maritime et fenêtres d’escale', 'lucide:ship'],
  [
    paths.simulation,
    'Simulation',
    'Comparaison de scénarios opérationnels',
    'lucide:flask-conical',
  ],
  [
    paths.quality,
    'Qualité des données',
    'Fraîcheur, complétude et fiabilité',
    'lucide:database-zap',
  ],
  [paths.audit, 'Journal d’audit', 'Traçabilité immuable des opérations', 'lucide:scroll-text'],
  [paths.reports, 'Rapports', 'Passage de quart et exports gouvernés', 'lucide:file-chart-column'],
] as const;

const Topbar = () => {
  const location = useLocation();
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 30_000);
    return () => window.clearInterval(timer);
  }, []);

  const definition = pageDefinitions.find(([path]) => location.pathname.startsWith(path));
  const page = location.pathname.startsWith(paths.capacity)
    ? {
        title: 'Escales & capacité',
        context: 'Priorisation temporelle des revues',
        icon: 'lucide:calendar-range',
        mode: 'HISTORIQUE D’EXERCICE',
        modeColor: pf.functional.amber,
      }
    : location.pathname.startsWith(paths.weather)
      ? {
          title: 'Météo & état de mer',
          context: 'Conditions, prévisions et vigilance',
          icon: 'lucide:cloud-sun',
          mode: 'SOURCES ACTIVES',
          modeColor: pf.functional.blue,
        }
      : {
          title: 'Tanger Med · Supervision',
          context: definition?.[1] ? `Poste décisionnel · ${definition[1]}` : 'Poste décisionnel',
          icon: definition?.[3] ?? 'lucide:anchor',
          mode: 'DONNÉES D’EXERCICE',
          modeColor: pf.functional.amber,
        };

  const localDate = useMemo(
    () =>
      new Intl.DateTimeFormat('fr-FR', {
        weekday: 'short',
        day: '2-digit',
        month: 'short',
        timeZone: 'Africa/Casablanca',
      }).format(now),
    [now],
  );
  const localTime = useMemo(
    () =>
      new Intl.DateTimeFormat('fr-FR', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        timeZone: 'Africa/Casablanca',
      }).format(now),
    [now],
  );

  return (
    <ElevationScroll>
      <AppBar
        position="fixed"
        sx={{
          left: { xs: 0, md: 78 },
          width: { xs: 1, md: 'calc(100% - 78px)' },
          borderBottom: `1px solid ${pf.structure.border}`,
          boxShadow: 'none',
          bgcolor: 'rgba(6,20,29,.92)',
          backdropFilter: 'blur(14px)',
          '@supports not (backdrop-filter: blur(14px))': { bgcolor: pf.background.navigation },
        }}
      >
        <Toolbar sx={{ minHeight: '72px !important', gap: 1.25, px: { xs: 1.5, sm: 2.5 } }}>
          <Box
            sx={{
              display: { xs: 'grid', md: 'none' },
              width: 36,
              height: 36,
              placeItems: 'center',
              color: pf.functional.cyan,
              bgcolor: pf.functional.cyanSoft,
              border: `1px solid ${pf.functional.cyan}45`,
              borderRadius: '8px',
            }}
          >
            <IconifyIcon icon="lucide:anchor" sx={{ fontSize: 18 }} />
          </Box>

          <Stack direction="row" alignItems="center" gap={1} minWidth={0}>
            <Box
              sx={{
                display: { xs: 'none', sm: 'grid' },
                width: 34,
                height: 34,
                placeItems: 'center',
                color: pf.functional.cyan,
                bgcolor: pf.background.panelRaised,
                border: `1px solid ${pf.structure.border}`,
                borderRadius: '8px',
              }}
            >
              <IconifyIcon icon={page.icon} sx={{ fontSize: 17 }} />
            </Box>
            <Box minWidth={0}>
              <Typography
                noWrap
                sx={{ color: pf.text.primary, fontSize: 14, lineHeight: '19px', fontWeight: 650 }}
              >
                {page.title}
              </Typography>
              <Typography
                noWrap
                sx={{
                  display: { xs: 'none', sm: 'block' },
                  color: pf.text.tertiary,
                  fontSize: 11,
                  lineHeight: '16px',
                }}
              >
                {page.context}
              </Typography>
            </Box>
          </Stack>

          <Stack direction="row" alignItems="center" gap={0.85} ml="auto">
            <ControlTowerActions />
            <Chip
              icon={<IconifyIcon icon="lucide:radio" />}
              label={page.mode}
              size="small"
              sx={{
                display: { xs: 'none', sm: 'flex' },
                height: 28,
                color: page.modeColor,
                bgcolor: `${page.modeColor}14`,
                border: `1px solid ${page.modeColor}45`,
                '& .MuiChip-icon': { color: page.modeColor },
              }}
            />
            <Box sx={{ display: { xs: 'none', sm: 'block' }, textAlign: 'right', minWidth: 92 }}>
              <Typography
                sx={{ color: pf.text.primary, fontSize: 12, lineHeight: '16px', fontWeight: 600 }}
              >
                {localTime}
              </Typography>
              <Typography
                sx={{
                  color: pf.text.tertiary,
                  fontSize: 11,
                  lineHeight: '15px',
                  textTransform: 'capitalize',
                }}
              >
                {localDate}
              </Typography>
            </Box>
          </Stack>
        </Toolbar>
      </AppBar>
    </ElevationScroll>
  );
};

export default Topbar;
