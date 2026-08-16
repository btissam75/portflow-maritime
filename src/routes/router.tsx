import { Suspense, lazy } from 'react';
import { Navigate, Outlet, createBrowserRouter } from 'react-router-dom';
import { rootPaths } from './paths';
import paths from './paths';

const App = lazy(() => import('App'));
const MainLayout = lazy(() => import('layouts/main-layout'));
const ControlTowerPage = lazy(() => import('pages/maritime/ControlTowerPage'));
const WeatherPage = lazy(() => import('pages/maritime/WeatherPage'));
const CapacityPage = lazy(() => import('pages/maritime/CapacityPage'));

import PageLoader from 'components/loading/PageLoader';
import Progress from 'components/loading/Progress';

export const routes = [
  {
    element: (
      <Suspense fallback={<Progress />}>
        <App />
      </Suspense>
    ),
    children: [
      {
        path: paths.controlTower,
        element: (
          <Suspense fallback={<PageLoader />}>
            <ControlTowerPage />
          </Suspense>
        ),
      },
      {
        path: rootPaths.root,
        element: (
          <MainLayout>
            <Suspense fallback={<PageLoader />}>
              <Outlet />
            </Suspense>
          </MainLayout>
        ),
        children: [
          {
            index: true,
            element: <Navigate to={paths.controlTower} replace />,
          },
          {
            path: paths.weather,
            element: <WeatherPage />,
          },
          {
            path: paths.capacity,
            element: <CapacityPage />,
          },
        ],
      },
      {
        path: '*',
        element: <Navigate to={paths.controlTower} replace />,
      },
    ],
  },
];

const router = createBrowserRouter(routes);

export default router;
