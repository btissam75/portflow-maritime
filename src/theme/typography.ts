import { TypographyOptions } from '@mui/material/styles/createTypography';

const bodyFont = [
  'Inter',
  '"Segoe UI Variable Text"',
  'Aptos',
  '"Segoe UI"',
  'Arial',
  'sans-serif',
].join(',');

const displayFont = bodyFont;

const headingFont = displayFont;

const typography: TypographyOptions = {
  fontFamily: bodyFont,
  h1: {
    fontFamily: displayFont,
    fontWeight: 700,
    fontSize: '1.75rem',
    lineHeight: '2.125rem',
  },
  h2: {
    fontFamily: displayFont,
    fontWeight: 700,
    fontSize: '1.5rem',
    lineHeight: '1.875rem',
  },
  h3: {
    fontFamily: headingFont,
    fontWeight: 700,
    fontSize: '1rem',
    lineHeight: '1.375rem',
  },
  h4: {
    fontFamily: headingFont,
    fontWeight: 600,
    fontSize: '1.1rem',
    lineHeight: 1.35,
  },
  h5: {
    fontFamily: headingFont,
    fontWeight: 600,
    fontSize: '1rem',
    lineHeight: 1.5,
  },
  h6: {
    fontFamily: headingFont,
    fontWeight: 600,
    fontSize: '0.9375rem',
    lineHeight: 1.5,
  },
  subtitle1: {
    fontWeight: 500,
    fontSize: '0.9375rem',
    lineHeight: 1.6,
  },
  subtitle2: {
    fontWeight: 500,
    fontSize: '0.8125rem',
    lineHeight: 1.5,
  },
  body1: {
    fontFamily: bodyFont,
    fontWeight: 400,
    fontSize: '0.9375rem',
    lineHeight: 1.55,
  },
  body2: {
    fontFamily: bodyFont,
    fontWeight: 400,
    fontSize: '0.8125rem',
    lineHeight: 1.5,
  },
  caption: {
    fontWeight: 600,
    fontSize: '0.7rem',
    lineHeight: 1.4,
  },
  button: {
    textTransform: 'none',
    fontFamily: bodyFont,
    fontWeight: 600,
    fontSize: '0.75rem',
    lineHeight: 1.25,
  },
};

export default typography;
