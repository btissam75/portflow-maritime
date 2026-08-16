import { Box } from '@mui/material';
import { PropsWithChildren } from 'react';
import { useLocation } from 'react-router-dom';
import NavigationRail from './NavigationRail';
import Topbar from './topbar/Topbar';

const MainLayout = ({ children }: PropsWithChildren) => {
  const location = useLocation();

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#06121C', background: 'radial-gradient(circle at 55% -10%, rgba(30,92,116,0.24) 0%, rgba(8,25,37,0.92) 35%, #06121C 72%)' }}>
      <NavigationRail />
      <Topbar />
      <Box
        component="main"
        sx={(theme) => ({
          flexGrow: 1,
          p: {
            xs: theme.spacing(2, 1.25, 10),
            sm: theme.spacing(2.5, 2, 10),
            md: theme.spacing(2.5, 2.5, 4),
            xl: theme.spacing(3, 3, 5),
          },
          ml: { xs: 0, md: '64px', xl: '208px' },
          pt: { xs: '78px', md: '82px' },
          minHeight: '100vh',
          width: { xs: 1, md: 'calc(100% - 64px)', xl: 'calc(100% - 208px)' },
          bgcolor: 'transparent',
        })}
      >
        <Box key={location.pathname} sx={{ animation: 'portflowRouteIn 280ms cubic-bezier(.2,.8,.2,1) both' }}>
          {children}
        </Box>
      </Box>
    </Box>
  );
};

export default MainLayout;
