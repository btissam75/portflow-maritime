import { PaletteColorOptions, PaletteOptions } from '@mui/material/styles';
import { grey, orange, red, green, blue, yellow } from './colors';

declare module '@mui/material/styles' {
  interface Palette {
    neutral: PaletteColor;
    green: PaletteColor;
    orange: PaletteColor;
    red: PaletteColor;
    yellow: PaletteColor;
  }

  interface PaletteOptions {
    neutral?: PaletteColorOptions;
    green?: PaletteColorOptions;
    orange?: PaletteColorOptions;
    red?: PaletteColorOptions;
    yellow?: PaletteColorOptions;
  }
  interface PaletteColor {
    lighter: string;
    darker: string;
  }
  interface SimplePaletteColorOptions {
    lighter?: string;
    darker?: string;
  }
}

const palette: PaletteOptions = {
  mode: 'dark',
  grey,
  text: {
    primary: '#EDF7F9',
    secondary: '#89A7B4',
    disabled: '#45616D',
  },

  background: {
    default: '#06121C',
    paper: '#0D2230',
  },

  action: {
    hover: 'rgba(54,214,207,0.07)',
    selected: 'rgba(54,214,207,0.13)',
  },

  neutral: {
    lighter: grey[50],
    light: grey[300],
    main: grey[500],
    dark: grey[700],
    darker: grey[900],
    contrastText: '#fff',
  },

  primary: {
    lighter: 'rgba(54,214,207,0.13)',
    light: '#6FE4DF',
    main: '#36D6CF',
    dark: '#1CA9A5',
    darker: '#0D5F61',
  },

  secondary: {
    lighter: 'rgba(73,167,255,0.14)',
    light: '#7AC0FF',
    main: '#49A7FF',
    dark: '#287CC7',
    darker: '#16466F',
  },

  error: {
    lighter: red[50],
    light: red[300],
    main: red[500],
    dark: red[700],
    darker: red[900],
  },

  warning: {
    lighter: orange[50],
    light: orange[300],
    main: orange[500],
    dark: orange[700],
    darker: orange[900],
  },

  success: {
    lighter: green[50],
    light: green[300],
    main: green[500],
    dark: green[700],
    darker: green[900],
  },

  info: {
    lighter: blue[50],
    light: blue[300],
    main: blue[500],
  },

  green: {
    light: green[100],
    main: green[200],
    dark: green[400],
    darker: green[600],
  },

  orange: {
    main: orange[400],
  },

  red: {
    main: red[800],
  },

  yellow: {
    main: yellow[500],
  },
};

export default palette;
