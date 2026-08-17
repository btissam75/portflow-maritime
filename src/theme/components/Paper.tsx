import { Theme } from '@mui/material';
import { Components } from '@mui/material/styles/components';

const Paper: Components<Omit<Theme, 'components'>>['MuiPaper'] = {
  defaultProps: {
    elevation: 1,
  },
  styleOverrides: {
    root: ({ theme }) => ({
      backgroundImage: 'none',
      backgroundColor: '#15191E',
      border: '1px solid #303840',
      borderRadius: theme.shape.borderRadius,
    }),
    elevation1: ({ theme }) => ({
      borderRadius: theme.shape.borderRadius,
      boxShadow: '0 16px 42px rgba(0,0,0,.24)',
    }),
  },
};

export default Paper;
