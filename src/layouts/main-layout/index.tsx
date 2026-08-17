import { Box } from '@mui/material';
import { PropsWithChildren } from 'react';
import { useLocation } from 'react-router-dom';
import NavigationRail from './NavigationRail';
import Topbar from './topbar/Topbar';

const MainLayout = ({ children }: PropsWithChildren) => {
  const location = useLocation();

  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: '#06141D',
        background:
          'radial-gradient(circle at 8% -10%, rgba(53,214,207,.08), transparent 30%), radial-gradient(circle at 92% 4%, rgba(61,137,177,.08), transparent 28%), linear-gradient(145deg, #06141D, #08202B 52%, #06131B)',
      }}
    >
      <NavigationRail />
      <Topbar />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          px: { xs: 1.25, sm: 2, md: 2.5, xl: 3 },
          pb: { xs: 10, md: 4, xl: 5 },
          ml: { xs: 0, md: '78px' },
          pt: { xs: '88px', md: '96px' },
          minHeight: '100vh',
          width: { xs: 1, md: 'calc(100% - 78px)' },
          bgcolor: 'transparent',
        }}
      >
        <Box
          key={location.pathname}
          sx={{ animation: 'portflowRouteIn 280ms cubic-bezier(.2,.8,.2,1) both' }}
        >
          {children}
        </Box>
      </Box>
    </Box>
  );
};

export default MainLayout;
