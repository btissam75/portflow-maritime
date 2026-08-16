import { CssBaseline, ThemeProvider } from '@mui/material';
import BreakpointsProvider from 'providers/BreakpointsProvider';
import { ReplayProvider } from 'providers/ReplayProvider';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider } from 'react-router-dom';
import router from 'routes/router';

import { theme } from 'theme/theme';
import './styles/maritime-colors.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <BreakpointsProvider>
        <CssBaseline />
        <ReplayProvider>
          <RouterProvider router={router} />
        </ReplayProvider>
      </BreakpointsProvider>
    </ThemeProvider>
  </React.StrictMode>,
);
