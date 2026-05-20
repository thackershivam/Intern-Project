/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      boxShadow: {
        glow: "0 0 24px rgba(14, 165, 233, 0.45)"
      }
    }
  },
  plugins: []
};
