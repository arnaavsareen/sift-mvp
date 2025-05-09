// frontend/tailwind.config.js
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#4A725D', // Main green
          50: '#EDF3F0',
          100: '#D7E4DB',
          200: '#B0C9BB',
          300: '#8AAE9C',
          400: '#6E9A85',
          500: '#4A725D', // Main green
          600: '#40624F',
          700: '#34503F',
          800: '#283D30',
          900: '#1C2920',
        },
        'green-main': '#4A725D',
        secondary: {
          DEFAULT: '#CEC5B0',
          50: '#FAF9F7',
          100: '#F5F3EF',
          200: '#ECE8DF',
          300: '#E3DECF',
          400: '#DAD3C0',
          500: '#CEC5B0', // Your secondary color
          600: '#B8AA8D',
          700: '#A38F6A',
          800: '#82714F',
          900: '#5F523A',
        },
        surface: {
          DEFAULT: '#243730',
          light: '#384842', // Your border color
          dark: '#1A2924',
        },
        background: '#2B3933', // Your background color
        border: '#384842', // Your border color
        light: '#F7E6CA', // Your light color
        success: '#4CAF50', // Your success color
        warning: '#FF9800', // Your warning color
        danger: '#F44336', // Your danger color
        info: '#2196F3', // Your info color
      },
      backgroundColor: {
        'dark': '#2B3933', // Your background color
        'darker': '#243730', // Your surface color
        'lighter': '#384842', // Your border color
      },
      textColor: {
        'dark-text': '#F7E6CA', // Your light color
        'dark-text-muted': '#CEC5B0', // Your secondary color
      },
      borderColor: {
        DEFAULT: '#384842', // Your border color
      },
    },
  },
  plugins: [],
}