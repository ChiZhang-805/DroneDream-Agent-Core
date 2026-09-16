import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 5173, strictPort: true },
  envPrefix: ["VITE_"],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("@supabase")) return "vendor-supabase";
          if (
            id.includes("react-markdown") ||
            id.includes("remark-") ||
            id.includes("rehype-") ||
            id.includes("micromark") ||
            id.includes("unified") ||
            id.includes("mdast-") ||
            id.includes("hast-")
          ) {
            return "vendor-markdown";
          }
          if (id.includes("@lobehub") || id.includes("lucide-react")) {
            return "vendor-icons";
          }
          return "vendor-core";
        },
      },
    },
  },
});
