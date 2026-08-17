import { Box } from '@mui/material';
import { useCallback, useEffect, useState } from 'react';
import MetoceanAnalyticsDashboard from 'components/sections/maritime/MetoceanAnalyticsDashboard';
import { liveMetoceanApi } from 'services/liveMetoceanApi';
import { metoceanApi } from 'services/metoceanApi';
import type { LiveMetoceanData } from 'types/liveMetocean';
import type { MetoceanDashboardData } from 'types/metocean';

const WeatherPage = () => {
  const [data, setData] = useState<MetoceanDashboardData | null>(null);
  const [liveData, setLiveData] = useState<LiveMetoceanData | null>(null);
  const [loading, setLoading] = useState(true);
  const [liveLoading, setLiveLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [outlookUpdatedAt, setOutlookUpdatedAt] = useState<string | null>(null);
  const [liveUpdatedAt, setLiveUpdatedAt] = useState<string | null>(null);

  const loadOutlook = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const response = await metoceanApi.getDashboard(signal);
      setData(response);
      const hasOutlookData = Boolean(
        response.status ||
          response.forecast.length ||
          response.impacts.length ||
          response.validation,
      );
      setOutlookUpdatedAt(hasOutlookData ? new Date().toISOString() : null);
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === 'AbortError') return;
      setError('Les prévisions à 72 h sont momentanément indisponibles.');
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  const loadLive = useCallback(async (signal?: AbortSignal) => {
    setLiveLoading(true);
    setLiveError(null);
    try {
      const response = await liveMetoceanApi.getDashboard(signal);
      setLiveData(response);
      setLiveUpdatedAt(response.atmosphere || response.marine ? response.fetchedAt : null);
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === 'AbortError') return;
      setLiveError('Les conditions actuelles sont momentanément indisponibles.');
    } finally {
      if (!signal?.aborted) setLiveLoading(false);
    }
  }, []);

  const loadAll = useCallback(
    async (signal?: AbortSignal) => {
      await Promise.allSettled([loadOutlook(signal), loadLive(signal)]);
    },
    [loadOutlook, loadLive],
  );

  useEffect(() => {
    const controller = new AbortController();
    void loadAll(controller.signal);
    return () => controller.abort();
  }, [loadAll]);

  useEffect(() => {
    if (!autoRefresh) return undefined;
    const interval = window.setInterval(() => void loadAll(), 5 * 60_000);
    return () => window.clearInterval(interval);
  }, [autoRefresh, loadAll]);

  return (
    <Box sx={{ width: 1, maxWidth: 1680, mx: 'auto' }}>
      <MetoceanAnalyticsDashboard
        data={data}
        liveData={liveData}
        loading={loading}
        liveLoading={liveLoading}
        outlookError={error}
        liveError={liveError}
        outlookUpdatedAt={outlookUpdatedAt}
        liveUpdatedAt={liveUpdatedAt}
        autoRefresh={autoRefresh}
        onAutoRefreshChange={setAutoRefresh}
        onRefresh={() => void loadAll()}
      />
    </Box>
  );
};

export default WeatherPage;
