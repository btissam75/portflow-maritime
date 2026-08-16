export const motionTokens = {
  duration: {
    instant: 80,
    fast: 140,
    base: 220,
    medium: 320,
    slow: 520,
    map: 800,
  },
  easing: {
    standard: 'cubic-bezier(0.2,0.8,0.2,1)',
    enter: 'cubic-bezier(0.16,1,0.3,1)',
    exit: 'cubic-bezier(0.4,0,1,1)',
  },
  spring: {
    primary: { stiffness: 320, damping: 32, mass: 0.8 },
    soft: { stiffness: 220, damping: 28, mass: 1 },
  },
} as const;
