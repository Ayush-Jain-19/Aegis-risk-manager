/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: {
      center: true,
      padding: "1.5rem",
    },
    extend: {
      colors: {
        // Editorial dark fintech surfaces — charcoal/slate, never flat black.
        canvas: "#14161B",
        surface: {
          DEFAULT: "#1B1E25",
          raised: "#21252D",
          sunken: "#171A20",
        },
        hairline: {
          DEFAULT: "#2A2F3A",
          strong: "#383F4D",
        },
        ink: {
          DEFAULT: "#ECEDF1",
          muted: "#8D93A3",
          faint: "#5B6272",
        },
        wire: {
          DEFAULT: "#7C93C4",
          dim: "#4C5A78",
        },
        signal: {
          approve: "#4FAE8A",
          "approve-bg": "#162420",
          "approve-border": "#28453A",
          review: "#D9A354",
          "review-bg": "#241E12",
          "review-border": "#453A20",
          block: "#DD6660",
          "block-bg": "#271717",
          "block-border": "#4A2727",
        },
      },
      fontFamily: {
        display: ["'Fraunces'", "ui-serif", "Georgia", "serif"],
        sans: ["'Inter'", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      borderRadius: {
        sm: "4px",
        DEFAULT: "6px",
        md: "8px",
        lg: "10px",
        xl: "14px",
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.02) inset, 0 12px 32px -16px rgba(0,0,0,0.6)",
        ring: "0 0 0 1px rgba(124,147,196,0.35)",
      },
      keyframes: {
        "pulse-dot": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
        "fade-up": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "pulse-dot": "pulse-dot 2s ease-in-out infinite",
        "fade-up": "fade-up 0.4s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
