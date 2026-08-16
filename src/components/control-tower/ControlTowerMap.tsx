import { Box, Button, IconButton, Slider, Stack, Tooltip, Typography } from '@mui/material';
import IconifyIcon from 'components/base/IconifyIcon';
import useReducedMotion from 'hooks/useReducedMotion';
import { useEffect, useMemo, useState } from 'react';
import { portflowPalette as pf } from 'theme/portflowPalette';

export interface TowerUnit {
  id: string;
  label: string;
  kind: 'vessel' | 'truck';
  x: number;
  y: number;
  status: 'normal' | 'watch' | 'critical';
  eta: string;
  zone: string;
}

export const towerUnits: TowerUnit[] = [
  { id: 'u1', label: 'NORDIC STAR', kind: 'vessel', x: 13, y: 30, status: 'normal', eta: '18 min', zone: 'Approche' },
  { id: 'u2', label: 'AL BORAN', kind: 'vessel', x: 28, y: 45, status: 'watch', eta: '34 min', zone: 'Couloir' },
  { id: 'u3', label: 'TIR 4821', kind: 'truck', x: 53, y: 34, status: 'normal', eta: '12 min', zone: 'ZRE' },
  { id: 'u4', label: 'TIR 7190', kind: 'truck', x: 68, y: 53, status: 'critical', eta: '46 min', zone: 'Scan' },
  { id: 'u5', label: 'ATLAS 08', kind: 'truck', x: 78, y: 73, status: 'normal', eta: '21 min', zone: 'Terminal' },
  { id: 'u6', label: 'MED LINK', kind: 'vessel', x: 35, y: 68, status: 'normal', eta: '52 min', zone: 'SAS' },
];

const statusColor = (status: TowerUnit['status']) => status === 'critical'
  ? pf.functional.red
  : status === 'watch' ? pf.functional.amber : pf.functional.green;

const zones = [
  { label: 'APPROCHE', color: '#2D91B8', left: '3%', top: '12%', width: '34%', height: '34%', clipPath: 'polygon(0 12%, 100% 0, 92% 100%, 12% 86%)', occupancy: 48 },
  { label: 'ZRE', color: '#2D91B8', left: '39%', top: '9%', width: '25%', height: '32%', clipPath: 'polygon(5% 0, 100% 8%, 90% 100%, 0 84%)', occupancy: 63 },
  { label: 'SCAN', color: '#508FDB', left: '65%', top: '12%', width: '28%', height: '31%', clipPath: 'polygon(8% 0, 100% 14%, 91% 100%, 0 82%)', occupancy: 82 },
  { label: 'PARK', color: '#237E85', left: '43%', top: '47%', width: '24%', height: '38%', clipPath: 'polygon(0 8%, 93% 0, 100% 92%, 10% 100%)', occupancy: 71 },
  { label: 'SAS', color: '#38A69A', left: '19%', top: '51%', width: '22%', height: '33%', clipPath: 'polygon(5% 0, 100% 12%, 91% 100%, 0 84%)', occupancy: 54 },
  { label: 'TERMINAL', color: '#2C7690', left: '69%', top: '49%', width: '27%', height: '39%', clipPath: 'polygon(0 7%, 95% 0, 100% 86%, 8% 100%)', occupancy: 76 },
];

interface ControlTowerMapProps {
  selectedId: string;
  onSelect: (id: string) => void;
}

const ControlTowerMap = ({ selectedId, onSelect }: ControlTowerMapProps) => {
  const reducedMotion = useReducedMotion();
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(5);
  const [progress, setProgress] = useState(64);
  const [heatmap, setHeatmap] = useState(true);
  const [traces, setTraces] = useState(true);

  useEffect(() => {
    if (!playing || reducedMotion) return undefined;
    const timer = window.setInterval(() => {
      if (document.hidden) return;
      setProgress((value) => value >= 100 ? 0 : Math.min(100, value + Math.max(1, speed / 5)));
    }, 700);
    return () => window.clearInterval(timer);
  }, [playing, reducedMotion, speed]);

  const selected = useMemo(() => towerUnits.find((unit) => unit.id === selectedId) ?? towerUnits[0], [selectedId]);

  return (
    <Box sx={{ position: 'relative', height: { xs: 430, lg: 474 }, overflow: 'hidden', bgcolor: pf.map.water }}>
      <Box
        sx={{
          position: 'absolute', inset: 0,
          backgroundImage: 'linear-gradient(rgba(53,98,116,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(53,98,116,0.10) 1px, transparent 1px)',
          backgroundSize: '32px 32px',
        }}
      />
      <Box sx={{ position: 'absolute', left: '7%', right: '5%', top: '15%', bottom: 70, bgcolor: pf.map.terrain, border: `1px solid ${pf.map.outline}`, clipPath: 'polygon(5% 0, 98% 5%, 100% 86%, 72% 100%, 4% 92%, 0 26%)' }} />

      {zones.map((zone) => (
        <Box
          key={zone.label}
          sx={{
            position: 'absolute', left: zone.left, top: zone.top, width: zone.width, height: zone.height,
            clipPath: zone.clipPath,
            bgcolor: heatmap ? `${zone.color}${Math.round(40 + zone.occupancy * 0.7).toString(16)}` : `${zone.color}35`,
            border: `1px solid ${zone.color}`,
            transition: reducedMotion ? 'opacity 120ms ease' : 'background-color 700ms cubic-bezier(0.2,0.8,0.2,1), transform 700ms cubic-bezier(0.2,0.8,0.2,1)',
          }}
        >
          <Stack alignItems="center" justifyContent="center" sx={{ height: 1 }}>
            <Typography sx={{ color: pf.text.primary, fontSize: 11, fontWeight: 700 }}>{zone.label}</Typography>
            <Typography sx={{ color: pf.map.label, fontSize: 12 }}>{zone.occupancy} %</Typography>
          </Stack>
        </Box>
      ))}

      {traces && (
        <>
          <Box sx={{ position: 'absolute', left: '13%', top: '34%', width: '46%', height: 2, bgcolor: pf.functional.cyan, opacity: 0.35, transform: 'rotate(7deg)', transformOrigin: 'left center' }} />
          <Box sx={{ position: 'absolute', left: '35%', top: '65%', width: '43%', height: 2, borderTop: `2px dashed ${pf.functional.blue}`, opacity: 0.48, transform: 'rotate(-9deg)', transformOrigin: 'left center' }} />
        </>
      )}

      {towerUnits.map((unit, index) => {
        const active = unit.id === selectedId;
        const color = active ? pf.functional.cyan : statusColor(unit.status);
        return (
          <Tooltip key={unit.id} title={`${unit.label} · ${unit.zone} · ETA ${unit.eta}`} arrow>
            <IconButton
              onClick={() => onSelect(unit.id)}
              aria-label={`Sélectionner ${unit.label}`}
              sx={{
                position: 'absolute', left: `${unit.x}%`, top: `${unit.y}%`, width: 32, height: 32,
                color, bgcolor: pf.background.navigation, border: `1px solid ${color}`,
                transform: active ? 'scale(1.12)' : 'scale(1)',
                transition: reducedMotion ? 'opacity 120ms ease' : 'transform 220ms cubic-bezier(0.16,1,0.3,1), border-color 140ms ease',
                animation: !reducedMotion && unit.status === 'critical' ? 'pfCriticalRing 2s ease-out 2' : 'none',
                '@keyframes pfCriticalRing': { '0%,100%': { boxShadow: '0 0 0 0 rgba(255,107,107,0)' }, '50%': { boxShadow: '0 0 0 8px rgba(255,107,107,0.16)' } },
                '&:hover': { bgcolor: pf.background.panelHover, transform: 'scale(1.12)' },
              }}
            >
              <IconifyIcon icon={unit.kind === 'vessel' ? 'lucide:ship' : 'lucide:truck'} sx={{ fontSize: 17 }} />
            </IconButton>
          </Tooltip>
        );
      })}

      <Stack direction="row" gap={0.6} sx={{ position: 'absolute', top: 12, left: 12 }}>
        <Button onClick={() => setHeatmap((value) => !value)} startIcon={<IconifyIcon icon="lucide:layers-3" />} sx={{ minHeight: 32, bgcolor: heatmap ? pf.functional.cyanSoft : pf.background.navigation, border: `1px solid ${heatmap ? pf.functional.cyan : pf.structure.border}`, color: heatmap ? pf.functional.cyan : pf.text.secondary, fontSize: 11 }}>
          Charge
        </Button>
        <Button onClick={() => setTraces((value) => !value)} startIcon={<IconifyIcon icon="lucide:route" />} sx={{ minHeight: 32, bgcolor: traces ? pf.functional.blueSoft : pf.background.navigation, border: `1px solid ${traces ? pf.functional.blue : pf.structure.border}`, color: traces ? pf.functional.blue : pf.text.secondary, fontSize: 11 }}>
          Trajectoires
        </Button>
      </Stack>

      <Box sx={{ position: 'absolute', top: 12, right: 12, width: 184, p: 1.1, bgcolor: 'rgba(7,23,34,0.92)', border: `1px solid ${pf.structure.border}`, borderRadius: '8px' }}>
        <Typography sx={{ color: pf.functional.cyan, fontSize: 11, fontWeight: 700 }}>UNITÉ SÉLECTIONNÉE</Typography>
        <Typography noWrap sx={{ color: pf.text.primary, fontSize: 13, fontWeight: 650, mt: 0.25 }}>{selected.label}</Typography>
        <Stack direction="row" justifyContent="space-between" mt={0.45}>
          <Typography sx={{ color: pf.text.secondary, fontSize: 12 }}>{selected.zone}</Typography>
          <Typography sx={{ color: statusColor(selected.status), fontSize: 12 }}>ETA {selected.eta}</Typography>
        </Stack>
      </Box>

      <Box sx={{ position: 'absolute', left: 12, right: 12, bottom: 10, px: 1.25, py: 0.8, bgcolor: 'rgba(7,23,34,0.94)', border: `1px solid ${pf.structure.border}`, borderRadius: '8px' }}>
        <Stack direction="row" alignItems="center" gap={1}>
          <IconButton onClick={() => setPlaying((value) => !value)} aria-label={playing ? 'Mettre le replay en pause' : 'Lire le replay'} sx={{ width: 32, height: 32, color: pf.background.primary, bgcolor: pf.functional.cyan, '&:hover': { bgcolor: '#54E3DD' } }}>
            <IconifyIcon icon={playing ? 'lucide:pause' : 'lucide:play'} sx={{ fontSize: 16 }} />
          </IconButton>
          <Typography sx={{ color: pf.text.secondary, fontSize: 11, whiteSpace: 'nowrap' }}>14:00</Typography>
          <Slider value={progress} onChange={(_, value) => setProgress(value as number)} aria-label="Position du replay" sx={{ color: pf.functional.cyan, mx: 0.5, '& .MuiSlider-thumb': { width: 12, height: 12 } }} />
          <Typography sx={{ color: pf.text.primary, fontSize: 11, whiteSpace: 'nowrap' }}>18:00</Typography>
          <Stack direction="row" gap={0.25}>
            {[1, 5, 20, 100].map((value) => (
              <Button key={value} onClick={() => setSpeed(value)} sx={{ minWidth: 34, minHeight: 28, px: 0.5, color: speed === value ? pf.background.primary : pf.text.secondary, bgcolor: speed === value ? pf.functional.cyan : 'transparent', fontSize: 11 }}>
                ×{value}
              </Button>
            ))}
          </Stack>
        </Stack>
      </Box>
    </Box>
  );
};

export default ControlTowerMap;
