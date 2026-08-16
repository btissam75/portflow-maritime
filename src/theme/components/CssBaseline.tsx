import { Theme } from '@mui/material';
import { Components } from '@mui/material/styles/components';
import scrollbar from 'theme/styles/scrollbar';
import echart from 'theme/styles/echart';

const CssBaseline: Components<Omit<Theme, 'components'>>['MuiCssBaseline'] = {
  styleOverrides: (theme) => ({
    body: {
      fontVariantLigatures: 'none',
      fontVariantNumeric: 'tabular-nums',
      letterSpacing: 0,
      color: '#EDF7F9',
      background: 'radial-gradient(circle at 55% -10%, rgba(30,92,116,0.24) 0%, rgba(8,25,37,0.92) 35%, #06121C 72%)',
      ...scrollbar(theme),
    },
    '::selection': {
      color: '#06121C',
      backgroundColor: '#36D6CF',
    },
    '@keyframes portflowPulse': {
      '0%, 100%': { transform: 'scale(1)', opacity: 1 },
      '50%': { transform: 'scale(1.65)', opacity: 0.45 },
    },
    '@keyframes portflowFadeUp': {
      from: { transform: 'translateY(8px)', opacity: 0 },
      to: { transform: 'translateY(0)', opacity: 1 },
    },
    '@keyframes portflowTitleIn': {
      from: { transform: 'translateY(18px)', opacity: 0, filter: 'blur(5px)' },
      to: { transform: 'translateY(0)', opacity: 1, filter: 'blur(0)' },
    },
    '@keyframes portflowTitleRule': {
      from: { transform: 'scaleX(0)', opacity: 0 },
      to: { transform: 'scaleX(1)', opacity: 1 },
    },
    '@keyframes portflowCardIn': {
      from: { transform: 'translateY(12px) scale(0.985)', opacity: 0 },
      to: { transform: 'translateY(0) scale(1)', opacity: 1 },
    },
    '@keyframes portflowSoftPulse': {
      '0%, 100%': { opacity: 0.72 },
      '50%': { opacity: 1 },
    },
    '@keyframes portflowSlideIn': {
      from: { transform: 'translateX(-10px)', opacity: 0 },
      to: { transform: 'translateX(0)', opacity: 1 },
    },
    '@keyframes portflowRouteIn': {
      from: { transform: 'translateY(10px)', opacity: 0, filter: 'blur(2px)' },
      to: { transform: 'translateY(0)', opacity: 1, filter: 'blur(0)' },
    },
    '@keyframes portflowStatusRing': {
      '0%': { transform: 'scale(0.7)', opacity: 0.9 },
      '75%, 100%': { transform: 'scale(1.7)', opacity: 0 },
    },
    '@keyframes portflowScan': {
      from: { transform: 'translateX(-100%)' },
      to: { transform: 'translateX(420%)' },
    },
    '@keyframes portflowRowIn': {
      from: { transform: 'translateX(-8px)', opacity: 0 },
      to: { transform: 'translateX(0)', opacity: 1 },
    },
    '@keyframes portflowSignalIn': {
      from: { transform: 'scaleX(0)', opacity: 0.35 },
      to: { transform: 'scaleX(1)', opacity: 1 },
    },
    '@media (prefers-reduced-motion: reduce)': {
      '*, *::before, *::after': {
        animationDuration: '0.01ms !important',
        animationIterationCount: '1 !important',
        scrollBehavior: 'auto !important',
        transitionDuration: '0.01ms !important',
      },
    },
    ...echart(),
  }),
};

export default CssBaseline;
