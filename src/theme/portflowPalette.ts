export const portflowPalette = {
  background: {
    primary: '#06121C',
    secondary: '#081925',
    navigation: '#071722',
    panel: '#0D2230',
    panelRaised: '#102937',
    panelHover: '#13313F',
  },
  structure: {
    border: '#1D3C4B',
    borderSoft: 'rgba(105,153,171,0.14)',
    divider: 'rgba(137,167,180,0.13)',
  },
  text: {
    primary: '#EDF7F9',
    secondary: '#89A7B4',
    tertiary: '#5F8190',
    disabled: '#45616D',
  },
  functional: {
    cyan: '#36D6CF',
    cyanSoft: 'rgba(54,214,207,0.13)',
    blue: '#49A7FF',
    blueSoft: 'rgba(73,167,255,0.14)',
    green: '#5DE0A3',
    greenSoft: 'rgba(93,224,163,0.13)',
    amber: '#FFBD4A',
    amberSoft: 'rgba(255,189,74,0.14)',
    red: '#FF6B6B',
    redSoft: 'rgba(255,107,107,0.14)',
    purple: '#A78BFA',
    purpleSoft: 'rgba(167,139,250,0.14)',
  },
  map: {
    water: '#082333',
    terrain: '#0A1D28',
    portZone: '#102F3E',
    road: '#294B59',
    secondaryRoad: '#1D3A47',
    outline: '#356274',
    label: '#A9C7D2',
  },
} as const;

export type PortflowPalette = typeof portflowPalette;
