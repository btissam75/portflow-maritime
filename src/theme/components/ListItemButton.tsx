import { Components, Theme } from '@mui/material';

const ListItemButton: Components<Omit<Theme, 'components'>>['MuiListItemButton'] = {
  styleOverrides: {
    gutters: ({ theme }) => ({
      borderRadius: theme.shape.borderRadius,
      transition: 'background-color 180ms ease, color 180ms ease, transform 180ms ease',
      '&:hover': {
        backgroundColor: '#E7F2F1',
        color: '#0B6667',
      },

      '&.Mui-selected': {
        backgroundColor: '#117C7D',
        color: theme.palette.common.white,
        boxShadow: 'none',
        '&:hover': {
          backgroundColor: '#0B6667',
        },
      },
    }),
  },
};
export default ListItemButton;
