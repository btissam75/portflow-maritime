import { Suspense, lazy } from 'react';
import { Navigate, Outlet, createBrowserRouter } from 'react-router-dom';
import { rootPaths } from './paths';
import paths from './paths';

const App = lazy(() => import('App'));
const MainLayout = lazy(() => import('layouts/main-layout'));
const WeatherPage = lazy(() => import('pages/maritime/WeatherPage'));
const CapacityPage = lazy(() => import('pages/maritime/CapacityPage'));
const ControlTowerPage = lazy(() => import('pages/control-tower/ControlTowerPage'));

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
            element: <Navigate to={paths.overview} replace />,
          },
          {
            path: paths.overview,
            element: <ControlTowerPage view="overview" />,
          },
          {
            path: paths.units,
            element: <ControlTowerPage view="units" />,
          },
          {
            path: paths.process,
            element: <ControlTowerPage view="process" />,
          },
          {
            path: paths.forecast,
            element: <ControlTowerPage view="forecast" />,
          },
          {
            path: paths.alerts,
            element: <ControlTowerPage view="alerts" />,
          },
          {
            path: paths.decisions,
            element: <ControlTowerPage view="decisions" />,
          },
          {
            path: paths.vessels,
            element: <ControlTowerPage view="vessels" />,
          },
          {
            path: paths.simulation,
            element: <ControlTowerPage view="simulation" />,
          },
          {
            path: paths.quality,
            element: <ControlTowerPage view="quality" />,
          },
          {
            path: paths.audit,
            element: <ControlTowerPage view="audit" />,
          },
          {
            path: paths.reports,
            element: <ControlTowerPage view="reports" />,
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
        element: <Navigate to={paths.overview} replace />,
      },
    ],
  },
];

const router = createBrowserRouter(routes);

export default router;
