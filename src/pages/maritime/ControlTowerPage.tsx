import {
  Avatar,
  Box,
  Button,
  Chip,
  IconButton,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  ThemeProvider,
  Tooltip,
  Typography,
} from '@mui/material';
import IconifyIcon from 'components/base/IconifyIcon';
import ControlTowerForecastChart from 'components/control-tower/ControlTowerForecastChart';
import ControlTowerMap, { towerUnits } from 'components/control-tower/ControlTowerMap';
import PortflowKpi from 'components/control-tower/PortflowKpi';
import PortflowPanel from 'components/control-tower/PortflowPanel';
import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import paths from 'routes/paths';
import { portflowPalette as pf } from 'theme/portflowPalette';
import { portflowShadows } from 'theme/portflowShadows';
import { portflowTheme } from 'theme/portflowTheme';

const navItems = [
  ['Control Tower', 'lucide:layout-dashboard'],
  ['Météo & état de mer', 'lucide:cloud-sun'],
  ['Escales & capacité', 'lucide:ship'],
  ['Flux des unités', 'lucide:route'],
  ['Prévisions IA', 'lucide:brain-circuit'],
  ['Scénarios', 'lucide:git-compare-arrows'],
  ['Incidents', 'lucide:triangle-alert'],
  ['Performance', 'lucide:chart-no-axes-combined'],
  ['Qualité des données', 'lucide:database-zap'],
] as const;

const decisionsSeed = [
  { id: 'd1', unitId: 'u4', priority: 'Critique', title: 'Renforcer le contrôle Scan', explanation: 'Saturation probable dans 42 min', confidence: 91, impact: '-18 min de file', color: pf.functional.red },
  { id: 'd2', unitId: 'u2', priority: 'Vigilance', title: 'Réserver le couloir B', explanation: 'Convergence de deux arrivées à H+1', confidence: 84, impact: '+12 % de fluidité', color: pf.functional.amber },
  { id: 'd3', unitId: 'u5', priority: 'Optimisation', title: 'Avancer le lot ATLAS 08', explanation: 'Fenêtre terminal disponible 28 min', confidence: 78, impact: '-9 min de cycle', color: pf.functional.cyan },
];

const stageLoads = [
  ['Approche', 48, pf.functional.cyan],
  ['ZRE', 63, pf.functional.blue],
  ['Scan', 82, pf.functional.amber],
  ['SAS', 54, pf.functional.cyan],
  ['Terminal', 76, pf.functional.blue],
] as const;

const tableRows = [
  ['NORDIC STAR', 'Navire', 'Approche', '18 min', 'Normal', '14:42'],
  ['AL BORAN', 'Navire', 'Couloir', '34 min', 'Vigilance', '14:41'],
  ['TIR 4821', 'TIR', 'ZRE', '12 min', 'Normal', '14:42'],
  ['TIR 7190', 'TIR', 'Scan', '46 min', 'Critique', '14:40'],
  ['ATLAS 08', 'TIR', 'Terminal', '21 min', 'Normal', '14:42'],
  ['MED LINK', 'Navire', 'SAS', '52 min', 'Normal', '14:39'],
] as const;

const statusColor = (status: string) => status === 'Critique'
  ? pf.functional.red
  : status === 'Vigilance' ? pf.functional.amber : pf.functional.green;

const ControlTowerPage = () => {
  const [selectedId, setSelectedId] = useState('u4');
  const [decisions, setDecisions] = useState(decisionsSeed);
  const [horizon, setHorizon] = useState(12);
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 30_000);
    return () => window.clearInterval(timer);
  }, []);

  const localTime = useMemo(() => new Intl.DateTimeFormat('fr-FR', {
    hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'Africa/Casablanca',
  }).format(now), [now]);
  const localDate = useMemo(() => new Intl.DateTimeFormat('fr-FR', {
    weekday: 'long', day: '2-digit', month: 'long', timeZone: 'Africa/Casablanca',
  }).format(now), [now]);

  const acknowledge = (decisionId: string, unitId: string) => {
    setSelectedId(unitId);
    setDecisions((items) => items.filter((item) => item.id !== decisionId));
  };

  return (
    <ThemeProvider theme={portflowTheme}>
      <Box
        sx={{
          minHeight: '100vh',
          color: pf.text.primary,
          bgcolor: pf.background.primary,
          background: 'radial-gradient(circle at 55% -10%, rgba(30,92,116,0.24) 0%, rgba(8,25,37,0.92) 35%, #06121C 72%)',
          fontFamily: 'Inter, "Segoe UI", Arial, sans-serif',
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        <Box
          component="aside"
          sx={{
            position: 'fixed', inset: '0 auto 0 0', zIndex: 50,
            width: { xs: 0, md: 64, xl: 208 }, overflow: 'hidden',
            bgcolor: pf.background.navigation, borderRight: `1px solid ${pf.structure.border}`,
            transition: 'width 220ms cubic-bezier(0.2,0.8,0.2,1)',
          }}
        >
          <Stack height="100%">
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
              {navItems.map(([label, icon], index) => {
                const active = index === 0;
                const navPath = index === 0 ? paths.controlTower : index === 1 ? paths.weather : index === 2 ? paths.capacity : null;
                return (
                  <Tooltip key={label} title={label} placement="right" disableHoverListener={false}>
                    <Button
                      component={Link}
                      to={navPath ?? paths.controlTower}
                      onClick={(event) => { if (!navPath) event.preventDefault(); }}
                      aria-disabled={!navPath}
                      startIcon={<IconifyIcon icon={icon} sx={{ fontSize: 17 }} />}
                      sx={{
                        position: 'relative', justifyContent: { md: 'center', xl: 'flex-start' }, minWidth: 0, height: 40,
                        px: { md: 0, xl: 1.15 }, color: active ? pf.text.primary : pf.text.secondary,
                        bgcolor: active ? pf.functional.cyanSoft : 'transparent',
                        background: active
                          ? 'linear-gradient(90deg, rgba(54,214,207,0.18), rgba(73,167,255,0.05))'
                          : 'transparent',
                        overflow: 'hidden',
                        '&::before': { content: '""', position: 'absolute', left: 0, width: 2, height: active ? 24 : 0, bgcolor: pf.functional.cyan },
                        '&:hover': { bgcolor: 'rgba(54,214,207,0.07)', color: pf.text.primary, transform: 'translateX(2px)' },
                        '& .MuiButton-startIcon': { m: { md: 0, xl: '0 8px 0 0' }, color: active ? pf.functional.cyan : 'inherit' },
                      }}
                    >
                      <Box component="span" sx={{ display: { md: 'none', xl: 'inline' }, fontSize: 13, whiteSpace: 'nowrap' }}>{label}</Box>
                    </Button>
                  </Tooltip>
                );
              })}
            </Stack>
            <Stack mt="auto" p={{ md: 1, xl: 1.5 }} gap={1}>
              <Button component={Link} to={paths.weather} startIcon={<IconifyIcon icon="lucide:cloud-sun" />} sx={{ justifyContent: { md: 'center', xl: 'flex-start' }, minWidth: 0, color: pf.text.secondary, border: `1px solid ${pf.structure.border}`, '& .MuiButton-startIcon': { m: { md: 0, xl: '0 8px 0 0' } } }}>
                <Box component="span" sx={{ display: { md: 'none', xl: 'inline' } }}>Météo & mer</Box>
              </Button>
              <Stack direction="row" alignItems="center" justifyContent={{ md: 'center', xl: 'flex-start' }} gap={1}>
                <Avatar sx={{ width: 34, height: 34, bgcolor: pf.functional.cyanSoft, color: pf.functional.cyan, fontSize: 12 }}>OP</Avatar>
                <Box sx={{ display: { md: 'none', xl: 'block' } }}>
                  <Typography sx={{ color: pf.text.primary, fontSize: 12, fontWeight: 600 }}>Opérateur salle</Typography>
                  <Typography sx={{ color: pf.text.tertiary, fontSize: 11 }}>Quart B</Typography>
                </Box>
              </Stack>
            </Stack>
          </Stack>
        </Box>

        <Box sx={{ ml: { xs: 0, md: '64px', xl: '208px' }, minWidth: 0 }}>
          <Stack
            component="header"
            direction="row"
            alignItems="center"
            justifyContent="space-between"
            sx={{ position: 'sticky', top: 0, zIndex: 40, height: 64, px: { xs: 1.5, md: 2.5 }, bgcolor: 'rgba(7,23,34,0.88)', backdropFilter: 'blur(14px)', borderBottom: `1px solid ${pf.structure.border}` }}
          >
            <Stack direction="row" alignItems="center" gap={1.1}>
              <Box sx={{ display: { xs: 'grid', md: 'none' }, width: 34, height: 34, placeItems: 'center', color: pf.functional.cyan, border: `1px solid ${pf.structure.border}`, borderRadius: '8px' }}>
                <IconifyIcon icon="lucide:anchor" />
              </Box>
              <Box>
                <Typography sx={{ color: pf.text.primary, fontSize: 14, fontWeight: 650 }}>Control Tower</Typography>
                <Typography sx={{ display: { xs: 'none', sm: 'block' }, color: pf.text.tertiary, fontSize: 11 }}>Supervision opérationnelle unifiée</Typography>
              </Box>
            </Stack>
            <Stack direction="row" alignItems="center" gap={1}>
              <Chip icon={<IconifyIcon icon="lucide:radio" />} label="LIVE" size="small" sx={{ height: 28, color: pf.functional.green, bgcolor: pf.functional.greenSoft, border: `1px solid ${pf.functional.green}45`, '& .MuiChip-icon': { color: pf.functional.green } }} />
              <Box sx={{ display: { xs: 'none', sm: 'block' }, textAlign: 'right' }}>
                <Typography sx={{ color: pf.text.primary, fontSize: 12, fontWeight: 600 }}>{localTime}</Typography>
                <Typography sx={{ color: pf.text.tertiary, fontSize: 11 }}>{localDate}</Typography>
              </Box>
              <IconButton aria-label="Notifications" sx={{ color: pf.text.secondary, border: `1px solid ${pf.structure.border}` }}>
                <IconifyIcon icon="lucide:bell" sx={{ fontSize: 17 }} />
              </IconButton>
            </Stack>
          </Stack>

          <Box component="main" sx={{ p: { xs: 1.25, sm: 2, xl: 2.5 }, pb: { xs: 8, md: 2.5 } }}>
            <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" alignItems={{ xs: 'flex-start', md: 'flex-end' }} gap={1} mb={1.5}>
              <Box>
                <Stack direction="row" alignItems="center" gap={0.8} mb={0.35}>
                  <Box sx={{ width: 22, height: 2, bgcolor: pf.functional.cyan }} />
                  <Typography sx={{ color: pf.functional.cyan, fontSize: 11, fontWeight: 600, textTransform: 'uppercase' }}>Tanger Med · Flux temps réel</Typography>
                </Stack>
                <Typography component="h1" sx={{ color: pf.text.primary, fontSize: { xs: 24, md: 28 }, lineHeight: '34px', fontWeight: 700 }}>
                  Supervision portuaire
                </Typography>
                <Typography sx={{ color: pf.text.secondary, fontSize: 13, lineHeight: '20px' }}>Situation, prévision et décisions sur un même espace opérationnel.</Typography>
              </Box>
              <Stack direction="row" gap={0.75}>
                <Box sx={{ px: 1.25, py: 0.65, bgcolor: pf.background.panel, border: `1px solid ${pf.structure.border}`, borderRadius: '8px' }}>
                  <Typography sx={{ color: pf.text.tertiary, fontSize: 11 }}>FRAÎCHEUR</Typography>
                  <Typography sx={{ color: pf.functional.green, fontSize: 12, fontWeight: 600 }}>12 s · Données nominales</Typography>
                </Box>
                <Button startIcon={<IconifyIcon icon="lucide:rotate-cw" />} sx={{ color: pf.background.primary, bgcolor: pf.functional.cyan, '&:hover': { bgcolor: '#54E3DD' } }}>Actualiser</Button>
              </Stack>
            </Stack>

            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2,minmax(0,1fr))', md: 'repeat(3,minmax(0,1fr))', xl: 'repeat(5,minmax(0,1fr))' }, gap: 1.5, mb: 1.5 }}>
              <PortflowKpi label="Arrivées prévues" value="38" unit="navires" detail="P10 32 · P90 45" icon="lucide:ship" delay={0} />
              <PortflowKpi label="Occupation terminal" value="72" unit="%" detail="+4 pts sur 2 heures" icon="lucide:gauge" accent={pf.functional.blue} delay={35} />
              <PortflowKpi label="Unités actives" value="1 264" detail="98 % localisées" icon="lucide:container" delay={70} />
              <PortflowKpi label="Décisions ouvertes" value={String(decisions.length)} detail="1 priorité critique" icon="lucide:clipboard-check" accent={decisions.length ? pf.functional.amber : pf.functional.green} delay={105} />
              <PortflowKpi label="Fiabilité ETA" value="87" unit="%" detail="+2,4 pts cette semaine" icon="lucide:circle-check-big" accent={pf.functional.green} delay={140} />
            </Box>

            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0,1fr)', lg: 'minmax(0,8fr) minmax(310px,4fr)' }, gap: 1.5, mb: 1.5 }}>
              <PortflowPanel
                title="Jumeau numérique du port"
                eyebrow="Carte opérationnelle"
                action={<Typography sx={{ color: pf.text.tertiary, fontSize: 11 }}>AIS + unités · {localTime}</Typography>}
              >
                <ControlTowerMap selectedId={selectedId} onSelect={setSelectedId} />
              </PortflowPanel>

              <PortflowPanel
                title="File de décisions"
                eyebrow="Priorisation IA"
                action={<Chip label={`${decisions.length} ouvertes`} size="small" sx={{ height: 25, color: pf.functional.amber, bgcolor: pf.functional.amberSoft }} />}
                sx={{ height: { xs: 'auto', lg: 522 } }}
              >
                <Box sx={{ maxHeight: { lg: 475 }, overflowY: 'auto' }}>
                  {decisions.length ? decisions.map((decision, index) => (
                    <Box key={decision.id} sx={{ px: 1.5, py: 1.35, borderBottom: `1px solid ${pf.structure.divider}`, bgcolor: decision.unitId === selectedId ? `${decision.color}09` : 'transparent', transition: 'background-color 220ms ease' }}>
                      <Stack direction="row" justifyContent="space-between" alignItems="center" gap={1}>
                        <Stack direction="row" alignItems="center" gap={0.75}>
                          <Typography sx={{ color: pf.text.tertiary, fontSize: 12, fontWeight: 600 }}>#{index + 1}</Typography>
                          <Chip label={decision.priority} size="small" sx={{ height: 23, color: decision.color, bgcolor: `${decision.color}18`, border: `1px solid ${decision.color}40`, fontSize: 11 }} />
                        </Stack>
                        <Typography sx={{ color: pf.text.secondary, fontSize: 11 }}>{decision.confidence} % confiance</Typography>
                      </Stack>
                      <Typography sx={{ color: pf.text.primary, fontSize: 13, lineHeight: '18px', fontWeight: 650, mt: 0.9 }}>{decision.title}</Typography>
                      <Typography sx={{ color: pf.text.secondary, fontSize: 12, lineHeight: '18px', mt: 0.35 }}>{decision.explanation}</Typography>
                      <Typography sx={{ color: pf.functional.green, fontSize: 12, mt: 0.55 }}>Impact estimé · {decision.impact}</Typography>
                      <Stack direction="row" gap={0.65} mt={1}>
                        <Button onClick={() => acknowledge(decision.id, decision.unitId)} sx={{ color: pf.background.primary, bgcolor: pf.functional.cyan, '&:hover': { bgcolor: '#54E3DD' } }}>Examiner</Button>
                        <Button onClick={() => setSelectedId(decision.unitId)} sx={{ color: pf.text.primary, bgcolor: pf.background.panelRaised, border: `1px solid ${pf.structure.border}` }}>Localiser</Button>
                      </Stack>
                    </Box>
                  )) : (
                    <Stack alignItems="center" justifyContent="center" gap={1} sx={{ minHeight: 300 }}>
                      <IconifyIcon icon="lucide:circle-check-big" sx={{ color: pf.functional.green, fontSize: 28 }} />
                      <Typography sx={{ color: pf.text.primary, fontSize: 13, fontWeight: 600 }}>Aucune décision en attente</Typography>
                      <Typography sx={{ color: pf.text.secondary, fontSize: 12 }}>La situation opérationnelle est sous contrôle.</Typography>
                    </Stack>
                  )}
                </Box>
              </PortflowPanel>
            </Box>

            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0,1fr)', lg: 'minmax(0,7fr) minmax(320px,5fr)' }, gap: 1.5, mb: 1.5 }}>
              <PortflowPanel
                title="Prévision probabiliste des flux"
                eyebrow="Réalisé et horizon"
                action={(
                  <Stack direction="row" gap={0.35}>
                    {[6, 12, 24].map((value) => <Button key={value} onClick={() => setHorizon(value)} sx={{ minWidth: 42, minHeight: 28, px: 0.7, color: horizon === value ? pf.background.primary : pf.text.secondary, bgcolor: horizon === value ? pf.functional.cyan : 'transparent' }}>H+{value}</Button>)}
                  </Stack>
                )}
              >
                <ControlTowerForecastChart height={238} />
              </PortflowPanel>

              <PortflowPanel title="Charge par étape" eyebrow={`Projection H+${horizon}`} action={<Typography sx={{ color: pf.text.tertiary, fontSize: 11 }}>Capacité nominale</Typography>}>
                <Stack gap={1.15} sx={{ p: 1.6, minHeight: 238, justifyContent: 'center' }}>
                  {stageLoads.map(([label, value, color]) => (
                    <Box key={label}>
                      <Stack direction="row" justifyContent="space-between" mb={0.45}>
                        <Typography sx={{ color: pf.text.primary, fontSize: 12, fontWeight: 600 }}>{label}</Typography>
                        <Typography sx={{ color: value >= 80 ? pf.functional.amber : pf.text.secondary, fontSize: 12 }}>{Math.min(96, value + Math.round((horizon - 12) / 4))} %</Typography>
                      </Stack>
                      <Box sx={{ height: 7, bgcolor: pf.background.secondary, borderRadius: '4px', overflow: 'hidden' }}>
                        <Box sx={{ width: `${Math.min(96, value + Math.round((horizon - 12) / 4))}%`, height: 1, bgcolor: value >= 80 ? pf.functional.amber : color, borderRadius: '4px', transition: 'width 520ms cubic-bezier(0.2,0.8,0.2,1)' }} />
                      </Box>
                    </Box>
                  ))}
                </Stack>
              </PortflowPanel>
            </Box>

            <PortflowPanel title="Unités supervisées" eyebrow="Vue consolidée" action={<Button endIcon={<IconifyIcon icon="lucide:sliders-horizontal" />} sx={{ color: pf.text.secondary }}>Filtres</Button>}>
              <Box sx={{ overflowX: 'auto' }}>
                <Table size="small" sx={{ minWidth: 760 }}>
                  <TableHead>
                    <TableRow sx={{ bgcolor: pf.background.secondary }}>
                      {['Unité', 'Type', 'Zone', 'Temps estimé', 'État', 'Dernier signal', ''].map((label) => <TableCell key={label} sx={{ color: pf.text.secondary, borderColor: pf.structure.divider, fontSize: 11, fontWeight: 600, textTransform: 'uppercase' }}>{label}</TableCell>)}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {tableRows.map((row, index) => (
                      <TableRow key={row[0]} hover selected={towerUnits[index]?.id === selectedId} sx={{ '&.Mui-selected': { bgcolor: pf.functional.cyanSoft }, '&:hover': { bgcolor: `${pf.functional.cyan}0A` } }}>
                        {row.map((value, cellIndex) => (
                          <TableCell key={`${row[0]}-${cellIndex}`} sx={{ color: cellIndex === 4 ? statusColor(value) : cellIndex === 0 ? pf.text.primary : pf.text.secondary, borderColor: pf.structure.divider, fontSize: 12, fontWeight: cellIndex === 0 ? 600 : 400 }}>{value}</TableCell>
                        ))}
                        <TableCell align="right" sx={{ borderColor: pf.structure.divider }}>
                          <IconButton onClick={() => setSelectedId(towerUnits[index]?.id ?? 'u1')} aria-label={`Localiser ${row[0]}`} sx={{ color: pf.functional.cyan }}><IconifyIcon icon="lucide:locate-fixed" sx={{ fontSize: 16 }} /></IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Box>
            </PortflowPanel>
          </Box>
        </Box>

        <Paper component="nav" sx={{ display: { xs: 'grid', md: 'none' }, position: 'fixed', left: 10, right: 10, bottom: 10, zIndex: 60, gridTemplateColumns: 'repeat(4,1fr)', bgcolor: 'rgba(7,23,34,0.96)', borderRadius: '8px', boxShadow: portflowShadows.raised, overflow: 'hidden' }}>
          {navItems.slice(0, 4).map(([label, icon], index) => (
            <Stack key={label} component={Link} to={index === 0 ? paths.controlTower : index === 1 ? paths.weather : index === 2 ? paths.capacity : paths.controlTower} alignItems="center" justifyContent="center" gap={0.25} sx={{ minHeight: 58, color: index === 0 ? pf.functional.cyan : pf.text.tertiary, borderTop: index === 0 ? `2px solid ${pf.functional.cyan}` : '2px solid transparent', textDecoration: 'none' }}>
              <IconifyIcon icon={icon} sx={{ fontSize: 17 }} />
              <Typography sx={{ color: 'inherit', fontSize: 11 }}>{index === 0 ? 'Contrôle' : label.split(' ')[0]}</Typography>
            </Stack>
          ))}
        </Paper>
      </Box>
    </ThemeProvider>
  );
};

export default ControlTowerPage;
