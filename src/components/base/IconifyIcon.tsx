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
  if (name.includes('boat')) {
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
  if (name.includes('notification')) {
    return (
      <>
        <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" />
        <path d="M10 21h4" />
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
    name.includes('timer')
  ) {
    return (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
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
