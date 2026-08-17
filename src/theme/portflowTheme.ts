import { createTheme } from '@mui/material';
import { motionTokens } from './motionTokens';
import { portflowPalette as pf } from './portflowPalette';
import { portflowShadows } from './portflowShadows';
import { portflowTypography } from './portflowTypography';

export const portflowTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: pf.functional.cyan },
    secondary: { main: pf.functional.blue },
    success: { main: pf.functional.green },
    warning: { main: pf.functional.amber },
    error: { main: pf.functional.red },
    background: { default: pf.background.primary, paper: pf.background.panel },
    text: { primary: pf.text.primary, secondary: pf.text.secondary, disabled: pf.text.disabled },
    divider: pf.structure.divider,
  },
  typography: {
    fontFamily: portflowTypography.fontFamily,
    h1: portflowTypography.display,
    h2: portflowTypography.pageTitle,
    h3: portflowTypography.sectionTitle,
    body1: portflowTypography.body,
    body2: portflowTypography.secondary,
    button: { fontSize: 12, lineHeight: 1.2, fontWeight: 650, textTransform: 'none' },
  },
  shape: { borderRadius: 12 },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        '*': { boxSizing: 'border-box' },
        body: { fontVariantNumeric: 'tabular-nums', letterSpacing: 0 },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: pf.background.panel,
          border: `1px solid ${pf.structure.border}`,
          boxShadow: portflowShadows.panel,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          minHeight: 34,
          borderRadius: 8,
          transition: `background-color ${motionTokens.duration.fast}ms ${motionTokens.easing.standard}, border-color ${motionTokens.duration.fast}ms ${motionTokens.easing.standard}, transform ${motionTokens.duration.fast}ms ${motionTokens.easing.standard}`,
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: pf.background.navigation,
          border: `1px solid ${pf.map.road}`,
          borderRadius: 8,
          boxShadow: portflowShadows.tooltip,
          color: pf.text.primary,
          fontSize: 12,
        },
      },
    },
  },
});
