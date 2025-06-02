/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    fontFamily: {
      sans: ['Samsung One', 'sans-serif'],
      serif: ['Samsung One', 'serif'],
      mono: ['Samsung One', 'monospace'],
      'samsung': ['Samsung One', 'sans-serif'],
    },
    extend: {
      fontWeight: {
        light: '300',
        normal: '400',
        semibold: '600',
        bold: '700',
        extrabold: '800',
      },
      animation: {
        'blob': 'blob 7s infinite',
        'blob-slow': 'blob 10s infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'particle': 'particle 4s ease-in-out infinite',
        'rotate-slow': 'rotate 8s linear infinite',
        'rotate-reverse': 'rotate-reverse 10s linear infinite',
        'float-slow': 'float 20s ease-in-out infinite',
      },
      keyframes: {
        blob: {
          '0%': {
            transform: 'translate(0px, 0px) scale(1)',
          },
          '33%': {
            transform: 'translate(30px, -50px) scale(1.1)',
          },
          '66%': {
            transform: 'translate(-20px, 20px) scale(0.9)',
          },
          '100%': {
            transform: 'translate(0px, 0px) scale(1)',
          },
        },
        float: {
          '0%, 100%': { transform: 'translate(0, 0) scale(1)' },
          '25%': { transform: 'translate(2%, 2%) scale(1.02)' },
          '50%': { transform: 'translate(-1%, 1%) scale(1.01)' },
          '75%': { transform: 'translate(1%, -1%) scale(1.03)' },
        },
        glow: {
          '0%': {
            opacity: 0.5,
            transform: 'scale(0.95)',
          },
          '100%': {
            opacity: 1,
            transform: 'scale(1.05)',
          },
        },
        particle: {
          '0%, 100%': {
            transform: 'translate(0, 0)',
            opacity: 0.5,
          },
          '50%': {
            transform: 'translate(10px, -10px)',
            opacity: 1,
          },
        },
        rotate: {
          '0%': { transform: 'translate(-50%, -50%) rotate(0deg)' },
          '100%': { transform: 'translate(-50%, -50%) rotate(360deg)' },
        },
        'rotate-reverse': {
          '0%': { transform: 'translate(-50%, -50%) rotate(0deg)' },
          '100%': { transform: 'translate(-50%, -50%) rotate(-360deg)' },
        },
      },
      backdropBlur: {
        35: '35px',
      },
    },
  },
  plugins: [],
}

