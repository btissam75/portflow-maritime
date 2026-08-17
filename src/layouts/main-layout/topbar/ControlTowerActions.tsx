import {
  Badge,
  Box,
  Button,
  Dialog,
  Divider,
  Drawer,
  IconButton,
  InputAdornment,
  ListItemButton,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import IconifyIcon from 'components/base/IconifyIcon';
import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import paths from 'routes/paths';
import { portflowPalette as pf } from 'theme/portflowPalette';

type Command = {
  label: string;
  detail: string;
  icon: string;
  keywords: string;
  to: string;
};

const commands: Command[] = [
  {
    label: 'Ouvrir la situation globale',
    detail: 'Flux, risques, capacités et décisions du quart',
    icon: 'lucide:layout-dashboard',
    keywords: 'dashboard global supervision control tower',
    to: paths.overview,
  },
  {
    label: 'Rechercher une unité prioritaire',
    detail: 'File de revue et fiche explicable',
    icon: 'lucide:container',
    keywords: 'unité camion conteneur eta risque fiche',
    to: paths.units,
  },
  {
    label: 'Surveiller le flux métier',
    detail: 'ZRE, Couloir, Park, Scan, PV, SAS et Terminal',
    icon: 'lucide:git-branch',
    keywords: 'process flux charge zone saturation',
    to: paths.process,
  },
  {
    label: 'Explorer les prévisions',
    detail: 'Charge, backlog et capacité de H+1 à H+24',
    icon: 'lucide:chart-no-axes-combined',
    keywords: 'prévision futur backlog capacité horizon',
    to: paths.forecast,
  },
  {
    label: 'Traiter les alertes',
    detail: 'Causes, impacts et recommandations',
    icon: 'lucide:shield-alert',
    keywords: 'alerte risque critique vigilance',
    to: paths.alerts,
  },
  {
    label: 'Piloter les décisions',
    detail: 'Affectation, exécution, vérification et clôture',
    icon: 'lucide:list-checks',
    keywords: 'décision kanban assigner action clôturer',
    to: paths.decisions,
  },
  {
    label: 'Simuler un scénario',
    detail: 'Renfort de capacité et règle de route',
    icon: 'lucide:flask-conical',
    keywords: 'simulation what if scénario capacité',
    to: paths.simulation,
  },
  {
    label: 'Voir les approches navires',
    detail: 'Carte maritime, ETA et unités associées',
    icon: 'lucide:ship',
    keywords: 'navire ais map carte eta escale',
    to: paths.vessels,
  },
  {
    label: 'Ouvrir la situation météo',
    detail: 'Conditions actuelles et carte du détroit',
    icon: 'lucide:cloud-sun',
    keywords: 'météo mer vagues vent pluie carte',
    to: paths.weather,
  },
  {
    label: 'Afficher la tendance à 72 heures',
    detail: 'Projection météo-marine longue échéance',
    icon: 'lucide:calendar-clock',
    keywords: 'météo projection 72 heures futur',
    to: `${paths.weather}?horizon=72`,
  },
  {
    label: 'Ouvrir la vigilance des escales',
    detail: 'File complète et capacité de revue',
    icon: 'lucide:ship',
    keywords: 'navire escale capacité liste',
    to: paths.capacity,
  },
  {
    label: 'Voir les escales critiques',
    detail: 'Priorités qui demandent une revue',
    icon: 'lucide:shield-alert',
    keywords: 'critique alerte risque retard priorité',
    to: `${paths.capacity}?filter=critical`,
  },
];

const ControlTowerActions = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [commandOpen, setCommandOpen] = useState(false);
  const [notificationOpen, setNotificationOpen] = useState(false);
  const [query, setQuery] = useState('');
  const isControlTower =
    !location.pathname.startsWith(paths.weather) && !location.pathname.startsWith(paths.capacity);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setCommandOpen(true);
      }
      if (event.key === 'Escape') setCommandOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  const visibleCommands = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase('fr-FR');
    if (!normalized) return commands;
    return commands.filter((command) =>
      `${command.label} ${command.detail} ${command.keywords}`
        .toLocaleLowerCase('fr-FR')
        .includes(normalized),
    );
  }, [query]);

  const controlTowerNotifications = [
    {
      title: 'Situation opérationnelle actualisée',
      detail: 'La file, les flux et les risques partagent le même instant de référence.',
      icon: 'lucide:radio',
      color: pf.functional.green,
    },
    {
      title: 'Trois alertes demandent une revue',
      detail: 'Les actions proposées sont disponibles dans le centre de décision.',
      icon: 'lucide:shield-alert',
      color: pf.functional.red,
    },
    {
      title: 'Moteurs métier à raccorder',
      detail: 'L’interface utilise actuellement le contrat dynamique d’exercice.',
      icon: 'lucide:plug-zap',
      color: pf.functional.amber,
    },
  ];

  const notifications = location.pathname.startsWith(paths.capacity)
    ? [
        {
          title: 'File de vigilance disponible',
          detail: 'Les priorités et la capacité de revue sont synchronisées.',
          icon: 'lucide:badge-check',
          color: pf.functional.green,
        },
        {
          title: 'Historique interactif prêt',
          detail: 'Comparez les six derniers cycles depuis la frise.',
          icon: 'lucide:history',
          color: pf.functional.blue,
        },
        {
          title: 'Validation opérateur requise',
          detail: 'Les brouillons restent locaux jusqu’à leur confirmation.',
          icon: 'lucide:user-check',
          color: pf.functional.amber,
        },
      ]
    : location.pathname.startsWith(paths.weather)
      ? [
          {
            title: 'Observations météo actives',
            detail: 'Les conditions atmosphériques et marines sont disponibles.',
            icon: 'lucide:radio',
            color: pf.functional.green,
          },
          {
            title: 'Tendance 72 h prête',
            detail: 'La projection peut être explorée depuis la frise.',
            icon: 'lucide:calendar-range',
            color: pf.functional.blue,
          },
          {
            title: 'Décision sous contrôle',
            detail: 'Les recommandations restent à confirmer par l’exploitation.',
            icon: 'lucide:shield-check',
            color: pf.functional.amber,
          },
        ]
      : controlTowerNotifications;

  const runCommand = (command: Command) => {
    navigate(command.to);
    setCommandOpen(false);
    setQuery('');
  };

  return (
    <>
      <Button
        onClick={() => setCommandOpen(true)}
        aria-label="Ouvrir la recherche globale"
        startIcon={<IconifyIcon icon="lucide:search" sx={{ fontSize: 16 }} />}
        sx={{
          display: { xs: 'none', lg: 'flex' },
          minHeight: 34,
          px: 1.15,
          color: pf.text.secondary,
          bgcolor: 'rgba(255,255,255,0.025)',
          border: `1px solid ${pf.structure.border}`,
          borderRadius: '8px',
          fontSize: 11,
          fontWeight: 500,
          textTransform: 'none',
          '&:hover': { color: pf.text.primary, bgcolor: pf.background.panelHover },
        }}
      >
        Rechercher
        <Box
          component="span"
          sx={{
            ml: 1.1,
            px: 0.6,
            py: 0.15,
            color: pf.text.tertiary,
            border: `1px solid ${pf.structure.border}`,
            borderRadius: '4px',
            fontSize: 9,
          }}
        >
          Ctrl K
        </Box>
      </Button>

      <Tooltip title="Centre de vigilance">
        <IconButton
          onClick={() => setNotificationOpen(true)}
          aria-label="Ouvrir le centre de vigilance"
          sx={{
            width: 36,
            height: 36,
            color: pf.text.secondary,
            border: `1px solid ${pf.structure.border}`,
            borderRadius: '8px',
            '&:hover': { color: pf.functional.cyan, bgcolor: pf.functional.cyanSoft },
          }}
        >
          <Badge
            badgeContent={notifications.length}
            sx={{ '& .MuiBadge-badge': { bgcolor: pf.functional.red, color: '#fff', fontSize: 8 } }}
          >
            <IconifyIcon icon="lucide:bell" sx={{ fontSize: 17 }} />
          </Badge>
        </IconButton>
      </Tooltip>

      {isControlTower && (
        <Button
          onClick={() => navigate(paths.simulation)}
          startIcon={<IconifyIcon icon="lucide:play" sx={{ fontSize: 16 }} />}
          sx={{
            display: { xs: 'none', md: 'flex' },
            minHeight: 36,
            px: 1.7,
            color: '#06141D',
            bgcolor: pf.functional.cyan,
            borderRadius: '10px',
            boxShadow: '0 10px 28px rgba(53,214,207,.16)',
            fontSize: 11,
            fontWeight: 850,
            textTransform: 'none',
            '&:hover': {
              bgcolor: '#74E7DE',
              boxShadow: '0 13px 34px rgba(53,214,207,.24)',
              transform: 'translateY(-1px)',
            },
          }}
        >
          Simuler une décision
        </Button>
      )}

      <Dialog
        open={commandOpen}
        onClose={() => setCommandOpen(false)}
        fullWidth
        maxWidth="sm"
        PaperProps={{
          sx: {
            mt: '-24vh',
            bgcolor: 'rgba(13,15,18,.98)',
            backgroundImage:
              'linear-gradient(145deg, rgba(85,214,194,.07), rgba(242,184,75,.025) 48%, transparent)',
            border: `1px solid ${pf.structure.border}`,
            borderRadius: '12px',
            boxShadow: '0 30px 90px rgba(0,0,0,0.52)',
            overflow: 'hidden',
          },
        }}
      >
        <TextField
          autoFocus
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Navire, météo, alerte, page…"
          aria-label="Rechercher une commande"
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <IconifyIcon icon="lucide:search" sx={{ color: pf.functional.cyan }} />
              </InputAdornment>
            ),
          }}
          inputProps={{ 'aria-label': 'Rechercher une commande' }}
          sx={{
            '& .MuiOutlinedInput-root': { minHeight: 60, color: pf.text.primary, fontSize: 15 },
            '& .MuiOutlinedInput-notchedOutline': { border: 0 },
          }}
        />
        <Divider sx={{ borderColor: pf.structure.border }} />
        <Stack p={0.8} gap={0.35}>
          {visibleCommands.map((command) => (
            <ListItemButton
              key={command.label}
              onClick={() => runCommand(command)}
              sx={{
                minHeight: 58,
                gap: 1.2,
                borderRadius: '8px',
                '&:hover': { bgcolor: pf.functional.cyanSoft },
              }}
            >
              <Box
                sx={{
                  width: 36,
                  height: 36,
                  display: 'grid',
                  placeItems: 'center',
                  color: pf.functional.cyan,
                  bgcolor: pf.functional.cyanSoft,
                  border: `1px solid ${pf.functional.cyan}35`,
                  borderRadius: '8px',
                }}
              >
                <IconifyIcon icon={command.icon} sx={{ fontSize: 18 }} />
              </Box>
              <Box minWidth={0}>
                <Typography sx={{ color: pf.text.primary, fontSize: 13, fontWeight: 650 }}>
                  {command.label}
                </Typography>
                <Typography sx={{ color: pf.text.tertiary, fontSize: 11 }}>
                  {command.detail}
                </Typography>
              </Box>
              <IconifyIcon
                icon="lucide:corner-down-left"
                sx={{ ml: 'auto', color: pf.text.tertiary, fontSize: 15 }}
              />
            </ListItemButton>
          ))}
          {!visibleCommands.length && (
            <Typography sx={{ py: 3, color: pf.text.tertiary, textAlign: 'center', fontSize: 12 }}>
              Aucune commande trouvée
            </Typography>
          )}
        </Stack>
      </Dialog>

      <Drawer
        anchor="right"
        open={notificationOpen}
        onClose={() => setNotificationOpen(false)}
        PaperProps={{
          sx: {
            width: { xs: '100%', sm: 390 },
            bgcolor: pf.background.navigation,
            backgroundImage:
              'radial-gradient(circle at 100% 0%, rgba(165,139,250,.10), transparent 38%)',
            borderLeft: `1px solid ${pf.structure.border}`,
          },
        }}
      >
        <Stack px={2.2} py={2} direction="row" alignItems="center" gap={1}>
          <Box
            sx={{
              width: 38,
              height: 38,
              display: 'grid',
              placeItems: 'center',
              color: pf.functional.cyan,
              bgcolor: pf.functional.cyanSoft,
              borderRadius: '9px',
            }}
          >
            <IconifyIcon icon="lucide:bell-ring" sx={{ fontSize: 19 }} />
          </Box>
          <Box>
            <Typography sx={{ color: pf.text.primary, fontSize: 16, fontWeight: 700 }}>
              Centre de vigilance
            </Typography>
            <Typography sx={{ color: pf.text.tertiary, fontSize: 11 }}>
              Informations utiles pour le quart en cours
            </Typography>
          </Box>
          <IconButton
            aria-label="Fermer le centre de vigilance"
            onClick={() => setNotificationOpen(false)}
            sx={{ ml: 'auto', color: pf.text.secondary }}
          >
            <IconifyIcon icon="lucide:x" />
          </IconButton>
        </Stack>
        <Divider sx={{ borderColor: pf.structure.border }} />
        <Stack p={1.4} gap={0.75}>
          {notifications.map((notification, index) => (
            <Box
              key={notification.title}
              sx={{
                p: 1.35,
                bgcolor: pf.background.panel,
                border: `1px solid ${pf.structure.border}`,
                borderRadius: '9px',
                animation: 'portflowSlideIn 360ms cubic-bezier(.2,.8,.2,1) both',
                animationDelay: `${index * 65}ms`,
              }}
            >
              <Stack direction="row" gap={1.1}>
                <Box
                  sx={{
                    width: 32,
                    height: 32,
                    display: 'grid',
                    placeItems: 'center',
                    color: notification.color,
                    bgcolor: `${notification.color}15`,
                    borderRadius: '7px',
                    flex: '0 0 auto',
                  }}
                >
                  <IconifyIcon icon={notification.icon} sx={{ fontSize: 16 }} />
                </Box>
                <Box>
                  <Typography sx={{ color: pf.text.primary, fontSize: 12.5, fontWeight: 650 }}>
                    {notification.title}
                  </Typography>
                  <Typography sx={{ color: pf.text.secondary, fontSize: 10.5, lineHeight: 1.5 }}>
                    {notification.detail}
                  </Typography>
                </Box>
              </Stack>
            </Box>
          ))}
        </Stack>
        <Typography sx={{ mt: 'auto', p: 2, color: pf.text.tertiary, fontSize: 10 }}>
          Les informations orientent la supervision. La décision reste confirmée par l’opérateur.
        </Typography>
      </Drawer>
    </>
  );
};

export default ControlTowerActions;
