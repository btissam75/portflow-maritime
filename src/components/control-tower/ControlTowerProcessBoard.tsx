import { Box, ButtonBase, Chip, Stack, Typography } from '@mui/material';
import IconifyIcon from 'components/base/IconifyIcon';
import type { TowerStage } from 'types/controlTower';
import { portflowPalette as pf } from 'theme/portflowPalette';

const stageTone = (occupancy: number) =>
  occupancy >= 95
    ? { color: pf.functional.red, label: 'SATURATION' }
    : occupancy >= 82
      ? { color: pf.functional.amber, label: 'SURVEILLANCE' }
      : { color: pf.functional.green, label: 'FLUIDE' };

const ControlTowerProcessBoard = ({
  stages,
  selected,
  onSelect,
  horizon = 3,
}: {
  stages: TowerStage[];
  selected?: string;
  onSelect: (stage: string) => void;
  horizon?: 1 | 3 | 6 | 12 | 24;
}) => (
  <Box
    component="section"
    aria-label="Trajectoire du flux métier"
    sx={{
      position: 'relative',
      overflowX: 'auto',
      py: 1.5,
      '@keyframes towerFlow': {
        from: { transform: 'translateX(-18px)', opacity: 0 },
        to: { transform: 'translateX(18px)', opacity: 1 },
      },
    }}
  >
    <Stack direction="row" alignItems="stretch" minWidth={1050} gap={0.65}>
      {stages.map((stage, index) => {
        const tone = stageTone(stage.occupancy_pct);
        const active = selected === stage.code;
        const projected = stage.forecast[`h${horizon}` as keyof TowerStage['forecast']];
        return (
          <Stack key={stage.code} direction="row" alignItems="center" flex={1} minWidth={0}>
            <ButtonBase
              onClick={() => onSelect(stage.code)}
              aria-pressed={active}
              sx={{
                width: 1,
                minHeight: 172,
                p: 1.25,
                display: 'block',
                textAlign: 'left',
                borderRadius: '16px',
                border: `1px solid ${active ? tone.color : pf.structure.border}`,
                background: `radial-gradient(circle at 85% 12%, ${tone.color}18, transparent 34%), linear-gradient(145deg, ${pf.background.panelRaised}, ${pf.background.panel})`,
                boxShadow: active
                  ? `0 18px 44px rgba(0,0,0,.3), 0 0 26px ${tone.color}12`
                  : '0 12px 30px rgba(0,0,0,.18)',
                transition: 'all 220ms ease',
                '&:hover': { transform: 'translateY(-4px)', borderColor: `${tone.color}88` },
              }}
            >
              <Stack direction="row" justifyContent="space-between" gap={0.5}>
                <Box>
                  <Typography sx={{ color: pf.text.tertiary, fontSize: 8, fontWeight: 900 }}>
                    ÉTAPE {String(index + 1).padStart(2, '0')}
                  </Typography>
                  <Typography
                    sx={{ color: pf.text.primary, fontSize: 14, fontWeight: 850, mt: 0.2 }}
                  >
                    {stage.label}
                  </Typography>
                </Box>
                <IconifyIcon
                  icon={
                    stage.code === 'TERMINAL'
                      ? 'lucide:warehouse'
                      : stage.code === 'SCAN'
                        ? 'lucide:scan-line'
                        : 'lucide:container'
                  }
                  sx={{ color: tone.color, fontSize: 20 }}
                />
              </Stack>
              <Stack direction="row" alignItems="baseline" gap={0.5} mt={1.1}>
                <Typography sx={{ color: tone.color, fontSize: 27, fontWeight: 850 }}>
                  {stage.units}
                </Typography>
                <Typography sx={{ color: pf.text.tertiary, fontSize: 9 }}>
                  / {stage.capacity}
                </Typography>
              </Stack>
              <Box
                sx={{
                  height: 5,
                  mt: 0.55,
                  bgcolor: pf.background.secondary,
                  borderRadius: 3,
                  overflow: 'hidden',
                }}
              >
                <Box
                  sx={{
                    width: `${Math.min(100, stage.occupancy_pct)}%`,
                    height: 1,
                    bgcolor: tone.color,
                    boxShadow: `0 0 10px ${tone.color}`,
                    transition: 'width 600ms ease',
                  }}
                />
              </Box>
              <Stack direction="row" justifyContent="space-between" mt={0.85}>
                <Typography sx={{ color: pf.text.secondary, fontSize: 8.5 }}>
                  P90 {stage.dwell_p90_h} h
                </Typography>
                <Typography sx={{ color: pf.functional.purple, fontSize: 8.5, fontWeight: 800 }}>
                  H+{horizon} · {projected}
                </Typography>
              </Stack>
              <Chip
                label={tone.label}
                size="small"
                sx={{
                  mt: 1,
                  height: 20,
                  color: tone.color,
                  bgcolor: `${tone.color}12`,
                  border: `1px solid ${tone.color}32`,
                  fontSize: 7.5,
                  fontWeight: 900,
                }}
              />
            </ButtonBase>
            {index < stages.length - 1 && (
              <Box
                sx={{
                  position: 'relative',
                  width: 22,
                  flexShrink: 0,
                  height: 2,
                  mx: 0.1,
                  bgcolor: pf.structure.border,
                }}
              >
                <Box
                  sx={{
                    position: 'absolute',
                    top: -2,
                    width: 6,
                    height: 6,
                    borderRadius: '50%',
                    bgcolor: pf.functional.cyan,
                    animation: 'towerFlow 1.8s linear infinite',
                  }}
                />
              </Box>
            )}
          </Stack>
        );
      })}
    </Stack>
    <Stack direction="row" gap={0.8} mt={1} alignItems="center">
      <Chip
        label="ROUTE DIRECTE"
        size="small"
        sx={{ color: pf.functional.green, bgcolor: pf.functional.greenSoft }}
      />
      <Typography sx={{ color: pf.text.tertiary, fontSize: 9 }}>
        Après Scan, le passage PV reste conditionnel ; la route directe rejoint le SAS.
      </Typography>
    </Stack>
  </Box>
);

export default ControlTowerProcessBoard;
