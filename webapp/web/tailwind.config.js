/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          900: "#04070a",
          800: "#070c10",
          700: "#0b1318",
          600: "#101c22",
        },
        line: "#173029",
        emerald: {
          DEFAULT: "#10b981",
          glow: "#34d399",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(16,185,129,0.25), 0 0 24px -6px rgba(16,185,129,0.35)",
      },
    },
  },
  plugins: [],
};
