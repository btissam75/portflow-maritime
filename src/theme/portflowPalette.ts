export const portflowPalette = {
  background: {
    primary: '#0B0D10',
    secondary: '#111419',
    navigation: '#0D0F12',
    panel: '#15191E',
    panelRaised: '#1B2026',
    panelHover: '#222930',
  },
  structure: {
    border: '#303840',
    borderSoft: 'rgba(213,206,190,0.11)',
    divider: 'rgba(215,219,223,0.10)',
  },
  text: {
    primary: '#F4F1EA',
    secondary: '#AAB1B6',
    tertiary: '#747D84',
    disabled: '#50575E',
  },
  functional: {
    cyan: '#55D6C2',
    cyanSoft: 'rgba(85,214,194,0.13)',
    blue: '#70A6E8',
    blueSoft: 'rgba(112,166,232,0.14)',
    green: '#6ED59B',
    greenSoft: 'rgba(110,213,155,0.13)',
    amber: '#F2B84B',
    amberSoft: 'rgba(242,184,75,0.14)',
    red: '#F46B68',
    redSoft: 'rgba(244,107,104,0.14)',
    purple: '#A58BFA',
    purpleSoft: 'rgba(165,139,250,0.14)',
  },
  map: {
    water: '#0A3150',
    terrain: '#6E5A45',
    portZone: '#8A704F',
    road: '#C2A36F',
    secondaryRoad: '#7C8890',
    outline: '#B29469',
    label: '#E7E0D3',
  },
} as const;

export type PortflowPalette = typeof portflowPalette;
