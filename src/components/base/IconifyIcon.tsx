import { Box, BoxProps } from '@mui/material';
import { Icon, IconProps } from '@iconify/react';
import type { ReactNode } from 'react';

interface IconifyProps extends BoxProps {
  icon: IconProps['icon'];
}

const localGlyph = (name: string): ReactNode => {
  if (name.includes('dashboard')) {
    return (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </>
    );
  }
  if (name.includes('container') || name.includes('truck')) {
    return name.includes('truck') ? (
      <>
        <path d="M3 7h11v10H3z" />
        <path d="M14 10h4l3 3v4h-7z" />
        <circle cx="7" cy="18" r="2" />
        <circle cx="18" cy="18" r="2" />
      </>
    ) : (
      <>
        <rect x="3" y="5" width="18" height="14" rx="2" />
        <path d="M7 5v14M17 5v14M10 9h4M10 15h4" />
      </>
    );
  }
  if (name.includes('git-branch') || name.includes('route') || name.includes('between')) {
    return (
      <>
        <circle cx="6" cy="5" r="2" />
        <circle cx="18" cy="7" r="2" />
        <circle cx="6" cy="19" r="2" />
        <path d="M6 7v10M8 12h4a6 6 0 0 0 6-3" />
      </>
    );
  }
  if (name.includes('chart')) {
    return (
      <>
        <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
        <path d="m3 8 6-5 6 7 6-5" />
      </>
    );
  }
  if (name.includes('list') || name.includes('scroll') || name.includes('file')) {
    return (
      <>
        <path d="M6 3h9l4 4v14H6z" />
        <path d="M15 3v5h5M9 12h7M9 16h7" />
      </>
    );
  }
  if (name.includes('flask')) {
    return (
      <>
        <path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4a2 2 0 0 0 1.8-3l-5-9V3" />
        <path d="M7.5 15h9" />
      </>
    );
  }
  if (name.includes('log-in')) {
    return (
      <>
        <path d="M14 4h6v16h-6" />
        <path d="M3 12h12M10 7l5 5-5 5" />
      </>
    );
  }
  if (name.includes('parking')) {
    return (
      <>
        <rect x="4" y="3" width="16" height="18" rx="3" />
        <path d="M9 17V7h4a3 3 0 0 1 0 6H9" />
      </>
    );
  }
  if (name.includes('scan')) {
    return (
      <>
        <path d="M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4" />
        <path d="M7 12h10" />
      </>
    );
  }
  if (name.includes('warehouse')) {
    return (
      <>
        <path d="m3 9 9-6 9 6v12H3z" />
        <path d="M7 21v-8h10v8M10 13v8M14 13v8" />
      </>
    );
  }
  if (
    name.includes('monitoring') ||
    name.includes('query-stats') ||
    name.includes('scatter') ||
    name.includes('timeline') ||
    name.includes('trending')
  ) {
    return (
      <>
        <path d="M4 19V5" />
        <path d="M4 19h16" />
        <path d="m7 15 4-4 3 2 5-6" />
      </>
    );
  }
  if (name.includes('boat') || name.includes('ship')) {
    return (
      <>
        <path d="M4 18h16l-2.5 3h-11z" />
        <path d="M7 18V9h10v9" />
        <path d="M10 9V5h4v4" />
        <path d="M3 22c1.2-1 2.4-1 3.6 0s2.4 1 3.6 0 2.4-1 3.6 0 2.4 1 3.6 0 2.4-1 3.6 0" />
      </>
    );
  }
  if (name.includes('anchor')) {
    return (
      <>
        <circle cx="12" cy="5" r="2" />
        <path d="M12 7v14" />
        <path d="M5 10h14" />
        <path d="M4 15a8 8 0 0 0 16 0" />
        <path d="m4 15 3 1" />
        <path d="m20 15-3 1" />
      </>
    );
  }
  if (name.includes('cloud-sun')) {
    return (
      <>
        <circle cx="16" cy="7" r="3" />
        <path d="M16 2v2M21 7h2M19.5 3.5 21 2" />
        <path d="M5 18h12a4 4 0 0 0 0-8 6 6 0 0 0-11.3 2A3 3 0 0 0 5 18Z" />
      </>
    );
  }
  if (name.includes('map')) {
    return (
      <>
        <path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3z" />
        <path d="M9 3v15" />
        <path d="M15 6v15" />
      </>
    );
  }
  if (name.includes('water') || name.includes('wave') || name.includes('air')) {
    return (
      <>
        <path d="M3 8c2.2 0 2.2-2 4.4-2s2.2 2 4.4 2 2.2-2 4.4-2 2.2 2 4.4 2" />
        <path d="M3 13c2.2 0 2.2-2 4.4-2s2.2 2 4.4 2 2.2-2 4.4-2 2.2 2 4.4 2" />
        <path d="M3 18c2.2 0 2.2-2 4.4-2s2.2 2 4.4 2 2.2-2 4.4-2 2.2 2 4.4 2" />
      </>
    );
  }
  if (name.includes('notification') || name.includes('bell')) {
    return (
      <>
        <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" />
        <path d="M10 21h4" />
      </>
    );
  }
  if (name.includes('radio')) {
    return (
      <>
        <circle cx="12" cy="12" r="2" />
        <path d="M8.5 8.5a5 5 0 0 0 0 7M15.5 8.5a5 5 0 0 1 0 7" />
        <path d="M5.5 5.5a9 9 0 0 0 0 13M18.5 5.5a9 9 0 0 1 0 13" />
      </>
    );
  }
  if (name.includes('warning') || name.includes('error')) {
    return (
      <>
        <path d="M12 3 2.8 20h18.4z" />
        <path d="M12 9v5" />
        <path d="M12 17h.01" />
      </>
    );
  }
  if (name.includes('experiment') || name.includes('model-training')) {
    return (
      <>
        <path d="M9 3h6" />
        <path d="M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4a2 2 0 0 0 1.8-3l-5-9V3" />
        <path d="M7.5 15h9" />
      </>
    );
  }
  if (
    name.includes('history') ||
    name.includes('schedule') ||
    name.includes('calendar') ||
    name.includes('timer') ||
    name.includes('event')
  ) {
    return (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    );
  }
  if (name.includes('skip-previous')) {
    return (
      <>
        <path d="M6 5v14" />
        <path d="m18 6-8 6 8 6z" />
      </>
    );
  }
  if (name.includes('skip-next')) {
    return (
      <>
        <path d="M18 5v14" />
        <path d="m6 6 8 6-8 6z" />
      </>
    );
  }
  if (name.includes('monitor-heart')) {
    return (
      <>
        <rect x="3" y="4" width="18" height="14" rx="2" />
        <path d="M7 12h2l2-4 3 7 2-3h2" />
        <path d="M9 21h6" />
      </>
    );
  }
  if (name.includes('tree')) {
    return (
      <>
        <rect x="3" y="3" width="6" height="5" rx="1" />
        <rect x="15" y="3" width="6" height="5" rx="1" />
        <rect x="9" y="16" width="6" height="5" rx="1" />
        <path d="M6 8v4h12V8" />
        <path d="M12 12v4" />
      </>
    );
  }
  if (name.includes('search') || name.includes('magnifier')) {
    return (
      <>
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-4-4" />
      </>
    );
  }
  if (name.includes('hamburger') || name.includes('menu')) {
    return (
      <>
        <path d="M4 7h16" />
        <path d="M4 12h16" />
        <path d="M4 17h16" />
      </>
    );
  }
  if (name.includes('pause')) {
    return (
      <>
        <path d="M9 5v14" />
        <path d="M15 5v14" />
      </>
    );
  }
  if (name.includes('play')) {
    return <path d="m8 5 11 7-11 7z" />;
  }
  if (name.includes('first-page')) {
    return (
      <>
        <path d="M6 5v14" />
        <path d="m18 6-8 6 8 6z" />
      </>
    );
  }
  if (name.includes('last-page')) {
    return (
      <>
        <path d="M18 5v14" />
        <path d="m6 6 8 6-8 6z" />
      </>
    );
  }
  if (
    name.includes('arrow') ||
    name.includes('chevron') ||
    name.includes('caret') ||
    name.includes('open-in-new') ||
    name.includes('expand')
  ) {
    const pointsLeft = name.includes('left') || name.includes('down');
    return pointsLeft ? (
      <>
        <path d="m15 18-6-6 6-6" />
        <path d="M9 12h10" />
      </>
    ) : (
      <>
        <path d="m9 18 6-6-6-6" />
        <path d="M5 12h10" />
      </>
    );
  }
  if (name.includes('person') || name.includes('account-box')) {
    return (
      <>
        <circle cx="12" cy="8" r="4" />
        <path d="M4 21a8 8 0 0 1 16 0" />
      </>
    );
  }
  if (name.includes('database')) {
    return (
      <>
        <ellipse cx="12" cy="5" rx="8" ry="3" />
        <path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5" />
        <path d="M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" />
      </>
    );
  }
  if (name.includes('refresh') || name.includes('rotate')) {
    return (
      <>
        <path d="M20 6v5h-5" />
        <path d="M4 18v-5h5" />
        <path d="M18.5 9A7 7 0 0 0 6.2 6.2L4 8M5.5 15A7 7 0 0 0 17.8 17.8L20 16" />
      </>
    );
  }
  if (name.includes('location')) {
    return (
      <>
        <path d="M20 10c0 5-8 11-8 11S4 15 4 10a8 8 0 1 1 16 0Z" />
        <circle cx="12" cy="10" r="2.5" />
      </>
    );
  }
  if (name.includes('thermostat')) {
    return (
      <>
        <path d="M14 14.8V5a3 3 0 0 0-6 0v9.8a5 5 0 1 0 6 0Z" />
        <path d="M11 7v9" />
      </>
    );
  }
  if (name.includes('speed')) {
    return (
      <>
        <path d="M4 18a8 8 0 1 1 16 0" />
        <path d="m12 14 4-4" />
        <path d="M7 18h10" />
      </>
    );
  }
  if (name.includes('bookmark')) {
    return <path d="M6 3h12v18l-6-4-6 4z" />;
  }
  if (name.includes('assignment')) {
    return (
      <>
        <rect x="5" y="4" width="14" height="17" rx="2" />
        <path d="M9 4V2h6v2M9 10h6M9 14h4" />
      </>
    );
  }
  if (
    name.includes('shield') ||
    name.includes('policy') ||
    name.includes('verified') ||
    name.includes('check') ||
    name.includes('done')
  ) {
    return (
      <>
        <path d="M12 3 5 6v5c0 5 3 8 7 10 4-2 7-5 7-10V6z" />
        <path d="m9 12 2 2 4-5" />
      </>
    );
  }
  if (name.includes('table')) {
    return (
      <>
        <rect x="3" y="4" width="18" height="16" rx="1" />
        <path d="M3 9h18" />
        <path d="M9 4v16" />
        <path d="M15 4v16" />
      </>
    );
  }
  if (name.includes('info')) {
    return (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v6" />
        <path d="M12 7h.01" />
      </>
    );
  }

  return (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M8 12h8" />
    </>
  );
};

const IconifyIcon = ({ icon, sx, ...rest }: IconifyProps) => {
  if (typeof icon !== 'string') {
    return <Box component={Icon} icon={icon} sx={sx} {...rest} />;
  }

  return (
    <Box
      component="svg"
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
      sx={{
        width: '1em',
        height: '1em',
        display: 'inline-block',
        flexShrink: 0,
        fill: 'none',
        stroke: 'currentColor',
        strokeWidth: 1.8,
        strokeLinecap: 'round',
        strokeLinejoin: 'round',
        ...sx,
      }}
      {...rest}
    >
      {localGlyph(icon)}
    </Box>
  );
};

export default IconifyIcon;
