import { Box, Chip, Stack, Typography } from '@mui/material';
import {
  AttributionControl,
  type GeoJSONSource,
  Map as MapLibreMap,
  Marker,
  NavigationControl,
} from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useEffect, useRef } from 'react';
import type { TowerStage, TowerUnit, TowerVessel } from 'types/controlTower';

const center: [number, number] = [-5.505, 35.888];

const zoneCenters: Record<string, [number, number]> = {
  ZRE: [-5.527, 35.897],
  COULOIR: [-5.519, 35.892],
  PARK: [-5.511, 35.888],
  SCAN: [-5.504, 35.884],
  PV: [-5.497, 35.889],
  SAS: [-5.49, 35.884],
  TERMINAL: [-5.478, 35.878],
};

const square = ([lng, lat]: [number, number], size = 0.0032) => [
  [lng - size, lat - size * 0.62],
  [lng + size, lat - size * 0.62],
  [lng + size, lat + size * 0.62],
  [lng - size, lat + size * 0.62],
  [lng - size, lat - size * 0.62],
];

const PortOperationsGIS = ({
  stages,
  units,
  vessels,
  selectedStage,
  onStageSelect,
}: {
  stages: TowerStage[];
  units: TowerUnit[];
  vessels: TowerVessel[];
  selectedStage?: string;
  onStageSelect: (stage: string) => void;
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new MapLibreMap({
      container: containerRef.current,
      center,
      zoom: 13.15,
      pitch: 58,
      bearing: -24,
      attributionControl: false,
      style: {
        version: 8,
        glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '© OpenStreetMap contributors',
          },
        },
        layers: [
          {
            id: 'osm',
            type: 'raster',
            source: 'osm',
            paint: {
              'raster-saturation': -0.62,
              'raster-brightness-max': 0.48,
              'raster-contrast': 0.22,
            },
          },
        ],
      },
    });
    map.addControl(new NavigationControl({ visualizePitch: true }), 'bottom-right');
    map.addControl(new AttributionControl({ compact: true }), 'bottom-left');

    map.on('load', () => {
      map.addSource('zones', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: stages.map((stage) => ({
            type: 'Feature',
            properties: {
              code: stage.code,
              label: stage.label,
              occupancy: stage.occupancy_pct,
              height: 45 + stage.occupancy_pct * 2.3,
            },
            geometry: { type: 'Polygon', coordinates: [square(zoneCenters[stage.code] ?? center)] },
          })),
        },
      });
      map.addLayer({
        id: 'zones-3d',
        type: 'fill-extrusion',
        source: 'zones',
        paint: {
          'fill-extrusion-color': [
            'case',
            ['>=', ['get', 'occupancy'], 95],
            '#ef476f',
            ['>=', ['get', 'occupancy'], 80],
            '#f5a524',
            '#22d3c5',
          ],
          'fill-extrusion-height': ['get', 'height'],
          'fill-extrusion-base': 3,
          'fill-extrusion-opacity': 0.68,
        },
      });
      map.addLayer({
        id: 'zones-label',
        type: 'symbol',
        source: 'zones',
        layout: {
          'text-field': [
            'concat',
            ['get', 'label'],
            '\n',
            ['to-string', ['round', ['get', 'occupancy']]],
            '%',
          ],
          'text-size': 12,
          'text-font': ['Open Sans Bold'],
          'text-offset': [0, -1.2],
        },
        paint: { 'text-color': '#f4fbff', 'text-halo-color': '#06131d', 'text-halo-width': 2 },
      });
      map.addSource('flow', {
        type: 'geojson',
        lineMetrics: true,
        data: {
          type: 'Feature',
          properties: {},
          geometry: {
            type: 'LineString',
            coordinates: ['ZRE', 'COULOIR', 'PARK', 'SCAN', 'SAS', 'TERMINAL'].map(
              (code) => zoneCenters[code],
            ),
          },
        },
      });
      map.addLayer({
        id: 'flow-glow',
        type: 'line',
        source: 'flow',
        paint: { 'line-color': '#32d7ff', 'line-width': 6, 'line-opacity': 0.2 },
      });
      map.addLayer({
        id: 'flow-line',
        type: 'line',
        source: 'flow',
        paint: { 'line-color': '#45e0d0', 'line-width': 2.5, 'line-dasharray': [2, 2] },
      });
      map.on('click', 'zones-3d', (event) => {
        const code = event.features?.[0]?.properties?.code as string | undefined;
        if (code) onStageSelect(code);
      });
      map.on('mouseenter', 'zones-3d', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mouseleave', 'zones-3d', () => {
        map.getCanvas().style.cursor = '';
      });

      vessels.slice(0, 5).forEach((vessel, index) => {
        const element = document.createElement('button');
        element.className = 'portflow-gis-vessel';
        element.title = `${vessel.name} · ETA ${new Date(vessel.predicted_eta).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}`;
        element.innerHTML = `<span>◆</span><b>${vessel.name}</b>`;
        new Marker({ element, anchor: 'center', rotation: vessel.heading })
          .setLngLat([center[0] - 0.034 - index * 0.008, center[1] - 0.005 + index * 0.008])
          .addTo(map);
      });
    });
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [onStageSelect, stages, vessels]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.isStyleLoaded() || !map.getSource('zones')) return;
    const source = map.getSource('zones') as GeoJSONSource;
    source.setData({
      type: 'FeatureCollection',
      features: stages.map((stage) => ({
        type: 'Feature',
        properties: {
          code: stage.code,
          label: stage.label,
          occupancy: stage.occupancy_pct,
          height: 45 + stage.occupancy_pct * 2.3,
        },
        geometry: { type: 'Polygon', coordinates: [square(zoneCenters[stage.code] ?? center)] },
      })),
    });
  }, [stages]);

  return (
    <Box
      sx={{
        position: 'relative',
        height: { xs: 430, lg: 560 },
        overflow: 'hidden',
        bgcolor: '#06131d',
      }}
    >
      <Box
        ref={containerRef}
        sx={{
          position: 'absolute',
          inset: 0,
          '& .maplibregl-canvas': { filter: 'sepia(.08) hue-rotate(154deg)' },
          '& .maplibregl-ctrl-group': {
            bgcolor: 'rgba(4,18,28,.9)',
            border: '1px solid rgba(105,151,168,.3)',
          },
          '& .maplibregl-ctrl-group button': { filter: 'invert(1)', opacity: 0.72 },
          '& .portflow-gis-vessel': {
            display: 'flex',
            alignItems: 'center',
            gap: '5px',
            color: '#69e6dc',
            bgcolor: 'rgba(4,18,28,.92)',
            border: '1px solid rgba(69,224,208,.5)',
            borderRadius: '5px',
            padding: '4px 7px',
            boxShadow: '0 8px 20px rgba(0,0,0,.35)',
            whiteSpace: 'nowrap',
            cursor: 'pointer',
            fontSize: '8px',
          },
          '& .portflow-gis-vessel span': { color: '#31d6ff', fontSize: '12px' },
          '& .portflow-gis-vessel b': { color: '#dcecf2', fontSize: '8px', letterSpacing: '.02em' },
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          background:
            'linear-gradient(90deg, rgba(3,12,20,.28), transparent 35%, rgba(3,12,20,.08)), linear-gradient(0deg, rgba(3,12,20,.45), transparent 42%)',
        }}
      />
      <Stack direction="row" gap={0.6} sx={{ position: 'absolute', top: 13, left: 13, zIndex: 4 }}>
        <Chip
          label="SIG · VUE 3D"
          size="small"
          sx={{
            color: '#7ee8dc',
            bgcolor: 'rgba(4,18,28,.88)',
            border: '1px solid rgba(69,224,208,.4)',
            fontWeight: 900,
          }}
        />
        <Chip
          label={`${units.length} unités suivies`}
          size="small"
          sx={{
            color: '#d9eaf0',
            bgcolor: 'rgba(4,18,28,.88)',
            border: '1px solid rgba(105,151,168,.35)',
          }}
        />
        {selectedStage && (
          <Chip
            label={`Filtre · ${selectedStage}`}
            size="small"
            onDelete={() => onStageSelect('')}
            sx={{ pointerEvents: 'auto', color: '#ffcc70', bgcolor: 'rgba(4,18,28,.92)' }}
          />
        )}
      </Stack>
      <Box
        sx={{
          position: 'absolute',
          right: 14,
          bottom: 14,
          zIndex: 3,
          px: 1.1,
          py: 0.75,
          bgcolor: 'rgba(4,18,28,.88)',
          border: '1px solid rgba(105,151,168,.3)',
          borderRadius: 1.5,
        }}
      >
        <Typography sx={{ color: '#d9eaf0', fontSize: 9, fontWeight: 850 }}>
          TANGER MED · COUCHE OPÉRATIONNELLE
        </Typography>
        <Typography sx={{ color: '#7897a4', fontSize: 7.5 }}>
          Zones 3D indicatives · fond © OpenStreetMap
        </Typography>
      </Box>
    </Box>
  );
};

export default PortOperationsGIS;
