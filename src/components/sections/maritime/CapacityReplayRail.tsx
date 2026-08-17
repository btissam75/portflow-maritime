import { Box, Button, Stack, Typography } from '@mui/material';
import IconifyIcon from 'components/base/IconifyIcon';
import type { CapacitySnapshot } from 'types/capacity';
import { portflowPalette as pf } from 'theme/portflowPalette';

const frameLabel = (value: string) =>
  new Intl.DateTimeFormat('fr-FR', {
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'Africa/Casablanca',
  }).format(new Date(value));

const CapacityReplayRail = ({
  snapshots,
  index,
  playing,
  speed,
  onIndexChange,
  onPlayingChange,
  onSpeedChange,
}: {
  snapshots: CapacitySnapshot[];
  index: number;
  playing: boolean;
  speed: 1 | 2 | 4;
  onIndexChange: (index: number) => void;
  onPlayingChange: (playing: boolean) => void;
  onSpeedChange: (speed: 1 | 2 | 4) => void;
}) => (
  <Box
    component="section"
    aria-label="Frise de l’historique des escales"
    sx={{
      px: { xs: 1.2, md: 1.7 },
      py: 1.15,
      bgcolor: pf.background.panel,
      border: `1px solid ${pf.structure.border}`,
      borderRadius: '8px',
      overflow: 'hidden',
    }}
  >
    <Stack direction={{ xs: 'column', md: 'row' }} alignItems={{ md: 'center' }} gap={1.3}>
      <Stack direction="row" alignItems="center" gap={0.65} minWidth={155}>
        <Button
          aria-label={playing ? 'Mettre la frise en pause' : 'Lire la frise historique'}
          onClick={() => onPlayingChange(!playing)}
          disabled={snapshots.length < 2}
          sx={{
            minWidth: 38,
            width: 38,
            height: 38,
            color: pf.background.primary,
            bgcolor: pf.functional.cyan,
            borderRadius: '50%',
            '&:hover': { bgcolor: '#5CE4DE' },
          }}
        >
          <IconifyIcon icon={playing ? 'lucide:pause' : 'lucide:play'} sx={{ fontSize: 16 }} />
        </Button>
        <Box>
          <Typography sx={{ color: pf.text.primary, fontSize: 10.5, fontWeight: 700 }}>
            Historique du quart
          </Typography>
          <Typography sx={{ color: pf.text.tertiary, fontSize: 9 }}>
            {snapshots.length ? `${index + 1} / ${snapshots.length} situations` : 'Préparation…'}
          </Typography>
        </Box>
      </Stack>

      <Box
        sx={{
          position: 'relative',
          display: 'grid',
          gridTemplateColumns: `repeat(${Math.max(snapshots.length, 1)},minmax(52px,1fr))`,
          alignItems: 'start',
          flex: 1,
          minWidth: 0,
          gap: 0.4,
          '&::before': {
            content: '""',
            position: 'absolute',
            left: 22,
            right: 22,
            top: 10,
            height: 2,
            bgcolor: pf.structure.border,
          },
        }}
      >
        {snapshots.map((snapshot, frameIndex) => {
          const active = frameIndex === index;
          const passed = frameIndex <= index;
          return (
            <Button
              key={snapshot.resolved_at}
              aria-pressed={active}
              aria-label={`Situation de ${frameLabel(snapshot.resolved_at)}`}
              onClick={() => onIndexChange(frameIndex)}
              sx={{
                zIndex: 1,
                minWidth: 0,
                minHeight: 45,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'flex-start',
                gap: 0.35,
                color: active ? pf.functional.cyan : pf.text.tertiary,
                fontSize: 8.5,
              }}
            >
              <Box
                sx={{
                  width: active ? 16 : 10,
                  height: active ? 16 : 10,
                  mt: active ? 0.2 : 0.55,
                  borderRadius: '50%',
                  bgcolor: passed ? pf.functional.cyan : pf.background.panelRaised,
                  border: `2px solid ${active ? '#fff' : pf.structure.border}`,
                  boxShadow: active ? `0 0 18px ${pf.functional.cyan}` : 'none',
                  transition: 'all 220ms ease',
                }}
              />
              {frameLabel(snapshot.resolved_at)}
            </Button>
          );
        })}
      </Box>

      <Stack direction="row" gap={0.35} alignItems="center">
        <Typography sx={{ color: pf.text.tertiary, fontSize: 9, mr: 0.25 }}>VITESSE</Typography>
        {([1, 2, 4] as const).map((value) => (
          <Button
            key={value}
            aria-pressed={speed === value}
            onClick={() => onSpeedChange(value)}
            sx={{
              minWidth: 34,
              minHeight: 32,
              color: speed === value ? pf.background.primary : pf.text.secondary,
              bgcolor: speed === value ? pf.functional.cyan : pf.background.secondary,
              border: `1px solid ${speed === value ? pf.functional.cyan : pf.structure.border}`,
              borderRadius: '5px',
              fontSize: 9,
            }}
          >
            ×{value}
          </Button>
        ))}
      </Stack>
    </Stack>
  </Box>
);

export default CapacityReplayRail;
