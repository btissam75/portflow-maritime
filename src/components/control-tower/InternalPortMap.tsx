import { Box, Button, ButtonBase, Chip, Stack, Typography } from '@mui/material';
import IconifyIcon from 'components/base/IconifyIcon';
import type { TowerStage, TowerUnit, TowerVessel } from 'types/controlTower';
import { portflowPalette as pf } from 'theme/portflowPalette';

const stageDesign: Record<
  string,
  {
    left: number;
    top: number;
    width: number;
    height: number;
    icon: string;
    color: string;
    detail: string;
  }
> = {
  ZRE: {
    left: 5,
    top: 18,
    width: 13,
    height: 15,
    icon: 'lucide:log-in',
    color: '#58C7F3',
    detail: 'Entrée régulée',
  },
  COULOIR: {
    left: 22,
    top: 15,
    width: 15,
    height: 17,
    icon: 'lucide:route',
    color: '#55D6C2',
    detail: 'Transit contrôlé',
  },
  PARK: {
    left: 5,
    top: 51,
    width: 18,
    height: 22,
    icon: 'lucide:square-parking',
    color: '#F2B84B',
    detail: 'Zone d’attente',
  },
  SCAN: {
    left: 29,
    top: 50,
    width: 15,
    height: 17,
    icon: 'lucide:scan-line',
    color: '#A58BFA',
    detail: 'Inspection scanner',
  },
  PV: {
    left: 48,
    top: 51,
    width: 14,
    height: 16,
    icon: 'lucide:shield-check',
    color: '#70A6E8',
    detail: 'Contrôle éventuel',
  },
  SAS: {
    left: 45,
    top: 16,
    width: 14,
    height: 17,
    icon: 'lucide:between-horizontal-end',
    color: '#6ED59B',
    detail: 'Pré-terminal',
  },
  TERMINAL: {
    left: 38,
    top: 78,
    width: 23,
    height: 15,
    icon: 'lucide:warehouse',
    color: '#55D6C2',
    detail: 'Zone d’embarquement',
  },
};

const tierColor = {
  CRITIQUE: pf.functional.red,
  VIGILANCE: pf.functional.amber,
  NORMAL: pf.functional.cyan,
} as const;

const InternalPortMap = ({
  stages,
  units,
  vessels = [],
  selectedStage,
  selectedUnit,
  onStageSelect,
  onUnitSelect,
  riskOnly,
  onRiskOnlyChange,
}: {
  stages: TowerStage[];
  units: TowerUnit[];
  vessels?: TowerVessel[];
  selectedStage?: string;
  selectedUnit?: string;
  onStageSelect: (stage: string) => void;
  onUnitSelect: (unit: string) => void;
  riskOnly: boolean;
  onRiskOnlyChange: (value: boolean) => void;
}) => {
  const visibleUnits = units
    .filter(
      (unit) =>
        (!riskOnly || unit.tier !== 'NORMAL') && (!selectedStage || unit.stage === selectedStage),
    )
    .slice(0, 31);

  const stageUnitIndex = new Map<string, number>();
  const positionedUnits = visibleUnits.map((unit) => {
    const design = stageDesign[unit.stage] ?? stageDesign.ZRE;
    const index = stageUnitIndex.get(unit.stage) ?? 0;
    stageUnitIndex.set(unit.stage, index + 1);
    const column = index % 5;
    const row = Math.floor(index / 5) % 3;
    return {
      unit,
      left: design.left + 1.7 + column * Math.max(1.7, (design.width - 4) / 5),
      top: design.top + design.height - 3.2 + row * 2.5,
    };
  });

  return (
    <Box
      component="section"
      aria-label="Jumeau numérique opérationnel du port"
      sx={{
        position: 'relative',
        minHeight: 555,
        overflow: 'auto hidden',
        bgcolor: '#061B25',
        border: '1px solid #174453',
        borderRadius: '0 0 14px 14px',
        '&::-webkit-scrollbar': { height: 4 },
        '&::-webkit-scrollbar-thumb': { bgcolor: '#245566', borderRadius: 3 },
      }}
    >
      <Box
        sx={{
          position: 'relative',
          minWidth: 960,
          minHeight: 555,
          overflow: 'hidden',
          backgroundImage:
            'linear-gradient(rgba(68,128,148,.14) 1px, transparent 1px), linear-gradient(90deg, rgba(68,128,148,.14) 1px, transparent 1px)',
          backgroundSize: '52px 52px',
          '@keyframes portflowTwinDash': { to: { strokeDashoffset: -34 } },
          '@keyframes portflowUnitPulse': {
            '0%,100%': { transform: 'scale(1)', boxShadow: '0 0 0 0 rgba(242,184,75,.4)' },
            '50%': { transform: 'scale(1.12)', boxShadow: '0 0 0 7px rgba(242,184,75,0)' },
          },
        }}
      >
        <Box
          sx={{
            position: 'absolute',
            inset: '0 0 0 66%',
            bgcolor: '#092D3F',
            backgroundImage:
              'radial-gradient(circle at 70% 20%, rgba(85,214,194,.10), transparent 34%), repeating-linear-gradient(165deg, rgba(112,166,232,.055) 0 1px, transparent 1px 22px)',
          }}
        />

        <Box
          component="svg"
          viewBox="0 0 1000 555"
          preserveAspectRatio="none"
          sx={{ position: 'absolute', inset: 0, width: 1, height: 1, pointerEvents: 'none' }}
        >
          <defs>
            <marker
              id="portflow-arrow"
              markerWidth="10"
              markerHeight="10"
              refX="8"
              refY="3"
              orient="auto"
              markerUnits="strokeWidth"
            >
              <path d="M0,0 L0,6 L9,3 z" fill="#4DE1D2" />
            </marker>
            <filter id="portflow-glow">
              <feGaussianBlur stdDeviation="2.4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <path
            d="M658 -20 C670 95 645 160 620 215 C592 275 600 335 625 390 C651 447 635 505 606 580"
            fill="none"
            stroke="#3F7282"
            strokeWidth="4"
          />
          <path
            d="M130 188 L292 188 Q370 188 370 244 L370 304 Q370 337 414 356 L535 410 Q556 421 535 446 L500 485"
            fill="none"
            stroke="rgba(77,225,210,.25)"
            strokeWidth="8"
          />
          <path
            d="M130 188 L292 188 Q370 188 370 244 L370 304 Q370 337 414 356 L535 410 Q556 421 535 446 L500 485"
            fill="none"
            stroke="#4DE1D2"
            strokeWidth="3"
            strokeDasharray="12 9"
            strokeDashoffset="0"
            markerEnd="url(#portflow-arrow)"
            filter="url(#portflow-glow)"
            style={{ animation: 'portflowTwinDash 2.2s linear infinite' }}
          />
          <path
            d="M370 304 Q435 280 515 302"
            fill="none"
            stroke="#70A6E8"
            strokeWidth="2"
            strokeDasharray="8 8"
            markerEnd="url(#portflow-arrow)"
          />
          <path
            d="M515 302 Q544 257 520 187"
            fill="none"
            stroke="#70A6E8"
            strokeWidth="2"
            strokeDasharray="8 8"
          />
          {vessels.slice(0, 3).map((vessel, index) => (
            <path
              key={vessel.vessel_id}
              d={`M${910 - index * 24} ${100 + index * 150} Q790 ${130 + index * 115} ${650 - index * 5} ${170 + index * 112}`}
              fill="none"
              stroke="rgba(85,214,194,.58)"
              strokeWidth="2"
              strokeDasharray="7 8"
            />
          ))}
        </Box>

        <Stack
          direction="row"
          gap={0.7}
          sx={{ position: 'absolute', top: 14, left: 14, zIndex: 8 }}
        >
          <Chip
            label={`${units.length.toLocaleString('fr-FR')} unités suivies`}
            sx={{
              height: 30,
              color: pf.text.primary,
              bgcolor: 'rgba(6,20,29,.90)',
              border: '1px solid #245566',
              fontSize: 9,
              fontWeight: 800,
            }}
          />
          <Chip
            label={`${units.filter((unit) => unit.tier !== 'NORMAL').length} vigilances`}
            sx={{
              height: 30,
              color: pf.functional.amber,
              bgcolor: 'rgba(6,20,29,.90)',
              border: `1px solid ${pf.functional.amber}55`,
              fontSize: 9,
              fontWeight: 800,
            }}
          />
        </Stack>
        <Button
          aria-pressed={riskOnly}
          onClick={() => onRiskOnlyChange(!riskOnly)}
          startIcon={<IconifyIcon icon="lucide:shield-alert" />}
          sx={{
            position: 'absolute',
            top: 14,
            right: 14,
            zIndex: 8,
            minHeight: 30,
            color: riskOnly ? '#06141D' : pf.text.secondary,
            bgcolor: riskOnly ? pf.functional.amber : 'rgba(6,20,29,.9)',
            border: `1px solid ${riskOnly ? pf.functional.amber : '#245566'}`,
            borderRadius: '8px',
            fontSize: 8.5,
          }}
        >
          À surveiller
        </Button>

        {stages.map((stage) => {
          const design = stageDesign[stage.code];
          if (!design) return null;
          const active = selectedStage === stage.code;
          const overloaded = stage.occupancy_pct >= 95;
          const color = overloaded ? pf.functional.red : design.color;
          return (
            <ButtonBase
              key={stage.code}
              onClick={() => onStageSelect(active ? '' : stage.code)}
              aria-pressed={active}
              sx={{
                position: 'absolute',
                left: `${design.left}%`,
                top: `${design.top}%`,
                width: `${design.width}%`,
                height: `${design.height}%`,
                zIndex: 4,
                display: 'block',
                p: 1.1,
                textAlign: 'left',
                color: pf.text.primary,
                bgcolor: active ? `${color}28` : '#0C2B37',
                border: `2px solid ${active ? color : `${color}75`}`,
                borderRadius:
                  stage.code === 'PARK'
                    ? '10px 22px 10px 10px'
                    : stage.code === 'TERMINAL'
                      ? '2px'
                      : '9px',
                boxShadow: active
                  ? `0 0 28px ${color}28, inset 0 0 24px ${color}10`
                  : '0 12px 26px rgba(0,0,0,.18)',
                transition: 'all 180ms ease',
                '&:hover': {
                  transform: 'translateY(-3px)',
                  borderColor: color,
                  bgcolor: `${color}1d`,
                },
              }}
            >
              <Stack direction="row" alignItems="center" gap={0.65}>
                <IconifyIcon icon={design.icon} sx={{ color, fontSize: 17 }} />
                <Typography sx={{ color, fontSize: 10, fontWeight: 900 }}>
                  {stage.label.toUpperCase()}
                </Typography>
              </Stack>
              <Typography sx={{ color: pf.text.primary, fontSize: 16, fontWeight: 850, mt: 0.6 }}>
                {stage.units}
              </Typography>
              <Typography sx={{ color: pf.text.tertiary, fontSize: 7.5 }}>
                {design.detail} · {stage.occupancy_pct.toFixed(0)}%
              </Typography>
              {stage.code === 'PARK' && (
                <Box
                  sx={{
                    mt: 0.5,
                    height: 9,
                    background:
                      'repeating-linear-gradient(90deg, transparent 0 12px, rgba(242,184,75,.35) 12px 14px)',
                  }}
                />
              )}
              {stage.code === 'SCAN' && (
                <Box
                  sx={{
                    position: 'absolute',
                    inset: '12% 10%',
                    borderLeft: `1px dashed ${color}55`,
                    borderRight: `1px dashed ${color}55`,
                    pointerEvents: 'none',
                  }}
                />
              )}
            </ButtonBase>
          );
        })}

        {positionedUnits.map(({ unit, left, top }) => {
          const active = selectedUnit === unit.unit_id;
          const color = tierColor[unit.tier];
          return (
            <Button
              key={unit.unit_id}
              aria-label={`${unit.unit_id}, ${unit.stage_label}, ${unit.tier}`}
              onClick={() => onUnitSelect(unit.unit_id)}
              sx={{
                position: 'absolute',
                left: `${left}%`,
                top: `${top}%`,
                zIndex: active ? 7 : 6,
                width: active ? 27 : 20,
                minWidth: active ? 27 : 20,
                height: active ? 27 : 20,
                minHeight: active ? 27 : 20,
                p: 0,
                color: '#06141D',
                bgcolor: color,
                border: `2px solid ${active ? '#fff' : '#061B25'}`,
                borderRadius: '6px',
                boxShadow: `0 0 ${active ? 20 : 8}px ${color}75`,
                animation:
                  unit.tier === 'CRITIQUE' ? 'portflowUnitPulse 1.8s ease-in-out infinite' : 'none',
                '&:hover': { transform: 'scale(1.25)' },
              }}
            >
              <IconifyIcon icon="lucide:truck" sx={{ fontSize: active ? 15 : 11 }} />
            </Button>
          );
        })}

        {selectedUnit && (
          <Box
            sx={{
              position: 'absolute',
              left: '37%',
              top: '41%',
              zIndex: 9,
              px: 1,
              py: 0.65,
              color: pf.text.primary,
              bgcolor: '#123746',
              border: '1px solid #5C8795',
              borderRadius: '7px',
              boxShadow: '0 10px 24px rgba(0,0,0,.3)',
            }}
          >
            <Typography sx={{ fontSize: 9, fontWeight: 850 }}>{selectedUnit}</Typography>
          </Box>
        )}

        {vessels.slice(0, 3).map((vessel, index) => (
          <Stack
            key={vessel.vessel_id}
            direction="row"
            alignItems="center"
            gap={0.7}
            sx={{
              position: 'absolute',
              zIndex: 5,
              left: `${77 + index * 2}%`,
              top: `${14 + index * 27}%`,
            }}
          >
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                bgcolor: pf.functional.cyan,
                boxShadow: `0 0 14px ${pf.functional.cyan}`,
              }}
            />
            <Box>
              <Typography sx={{ color: pf.text.primary, fontSize: 8.5, fontWeight: 850 }}>
                {vessel.name.toUpperCase()}
              </Typography>
              <Typography sx={{ color: pf.functional.cyan, fontSize: 7.5 }}>
                ETA{' '}
                {new Intl.DateTimeFormat('fr-FR', {
                  hour: '2-digit',
                  minute: '2-digit',
                  timeZone: 'Africa/Casablanca',
                }).format(new Date(vessel.predicted_eta))}
              </Typography>
            </Box>
          </Stack>
        ))}

        <Stack
          direction="row"
          alignItems="center"
          gap={0.7}
          sx={{ position: 'absolute', left: '70%', bottom: 22, color: '#84A9B6' }}
        >
          <IconifyIcon icon="lucide:ship" sx={{ fontSize: 28 }} />
          <Typography sx={{ fontSize: 8.5 }}>APPROCHES MARITIMES</Typography>
        </Stack>
        <Typography
          sx={{ position: 'absolute', right: 14, bottom: 12, color: '#6E95A3', fontSize: 7.5 }}
        >
          Vue opérationnelle schématique · localisation par zone métier
        </Typography>
      </Box>
    </Box>
  );
};

export default InternalPortMap;
