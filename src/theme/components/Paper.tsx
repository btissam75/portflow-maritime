import { Theme } from '@mui/material';
import { Components } from '@mui/material/styles/components';

const Paper: Components<Omit<Theme, 'components'>>['MuiPaper'] = {
  defaultProps: {
    elevation: 1,
  },
  styleOverrides: {
    root: ({ theme }) => ({
      backgroundImage: 'none',
      backgroundColor: '#0D2230',
      border: '1px solid #1D3C4B',
      borderRadius: theme.shape.borderRadius,
    }),
    elevation1: ({ theme }) => ({
      borderRadius: theme.shape.borderRadius,
      boxShadow: '0 10px 30px rgba(0,0,0,0.18)',
    }),
  },
};

export default Paper;
